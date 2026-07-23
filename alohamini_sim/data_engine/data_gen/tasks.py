"""
Simple, GPU-parallel tabletop tasks for the AlohaMini parallel-gripper robot.

These subclass ManiSkill's stock tabletop tasks (which already provide cube
spawning, goal sites, success/grasp checks) and just bind them to the
`aloha_mini_so100_v2` robot. They rely on the agent interface implemented in
maniskill/agents/aloha_mini/aloha_mini_so100_v2.py:
    - agent.tcp_pose          (left-arm TCP, midpoint of the two finger tips)
    - agent.is_grasping(obj)  (two-finger contact check)
    - agent.is_static()       (excludes the base joints)

NOTE on placement/reachability: the AlohaMini has a virtual mobile base
(root_x/y prismatic + root_z rotation) plus a vertical lift, so the table and
cube-spawn region usually need a small offset for the arm to reach. The
`cube_spawn_center` / agent spawn pose below are starting points to be tuned
against the scripted policy (see scripted_policy.py / generate.py).
"""

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

ROBOT = "aloha_mini_so100_v2"

# Geometry tuned to the AlohaMini left-arm workspace (gripper reaches z in ~[0.52, 1.28]
# with the base at x=-0.35); see maniskill/tools/ probes. The cube sits on a raised
# platform so the floor-standing robot can reach it like a real tabletop.
TABLE_TOP_Z = 0.70           # mid of the left-arm reachable z-range (~0.52..1.28)
CUBE_HALF = 0.02
CUBE_XY = (-0.13, -0.45)      # reachable spot, far enough that the pedestal clears the base
PEDESTAL_HALF = 0.02         # <= cube width so the open jaws clear it when grasping
LIFT_SUCCESS = 0.06          # cube must rise this far above the table to count
PICK_PLACE_GOAL_XY = (-0.13, -0.35)
STACK_BASE_XY = (-0.13, -0.35)
MULTI_YCB_DEFAULT_OBJECT_IDS = [
    "065-a_cups",
    "077_rubiks_cube",
    "012_strawberry",
    "058_golf_ball",
]
MULTI_TABLE_CENTER = (-0.13, -0.45)
MULTI_TABLE_HALF = (0.30, 0.24)
MULTI_TABLE_THICKNESS = 0.035
MULTI_FALLBACK_CUBE_HALF = 0.018

_YCB_INFO_CACHE: dict[str, Any] | None = None


@register_env("AlohaMiniTablePick-v1", max_episode_steps=120)
class AlohaMiniTablePickEnv(BaseEnv):
    """Pick a cube off a raised table with the AlohaMini left arm + parallel gripper.

    Designed for the floor-standing AlohaMini (gripper rest height ~1 m): the cube
    rests on a platform at z=0.50 so it is inside the left arm's reachable workspace.
    Success = the cube is grasped and lifted >= LIFT_SUCCESS above the table.
    """

    SUPPORTED_ROBOTS = [ROBOT]

    def __init__(self, *args, cube_xy_noise=0.0, **kwargs):
        self.cube_xy_noise = cube_xy_noise
        super().__init__(*args, robot_uids=ROBOT, **kwargs)

    @property
    def _default_sim_config(self):
        # More solver iterations + low bounce -> stable stiff grasp contacts so a
        # firmly-gripped cube can be lifted without the PhysX solver ejecting it.
        from mani_skill.utils.structs.types import SimConfig, SceneConfig
        return SimConfig(scene_config=SceneConfig(
            solver_position_iterations=30,
            solver_velocity_iterations=5,
            bounce_threshold=0.01,
        ))

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, -0.5, 0.9], target=CUBE_XY + (TABLE_TOP_Z,))
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.55, -0.75, 1.0],
                                    target=[CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z + 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.35, 0, 0]))

    def _load_scene(self, options: dict):
        # Small raised pedestal (static) whose top is at TABLE_TOP_Z. Kept small /
        # placed under the cube so the robot's mobile base does not collide with it.
        b = self.scene.create_actor_builder()
        half = [PEDESTAL_HALF, PEDESTAL_HALF, TABLE_TOP_Z / 2]
        b.add_box_collision(half_size=half)
        b.add_box_visual(half_size=half,
                         material=sapien.render.RenderMaterial(base_color=[0.6, 0.5, 0.4, 1]))
        b.initial_pose = sapien.Pose(p=[CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z / 2])
        self.platform = b.build_static(name="platform")
        # The cube to pick
        self.cube = actors.build_cube(
            self.scene, half_size=CUBE_HALF, color=[1, 0, 0, 1], name="cube",
            initial_pose=sapien.Pose(p=[CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z + CUBE_HALF]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # Robot to its 'ready' keyframe (arms up, grippers open)
            self.agent.reset(self.agent.keyframes["ready"].qpos)
            self.agent.robot.set_pose(sapien.Pose(p=[-0.35, 0, 0]))
            # Place the cube on the platform (optional small xy noise for variety)
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = CUBE_XY[0] + (torch.rand(b) * 2 - 1) * self.cube_xy_noise
            xyz[:, 1] = CUBE_XY[1] + (torch.rand(b) * 2 - 1) * self.cube_xy_noise
            xyz[:, 2] = TABLE_TOP_Z + CUBE_HALF
            self.cube.set_pose(Pose.create_from_pq(xyz))

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose, is_grasped=info["is_grasped"])
        if "state" in self.obs_mode:
            obs.update(obj_pose=self.cube.pose.raw_pose,
                       tcp_to_obj=self.cube.pose.p - self.agent.tcp_pose.p)
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.cube, arm_id=1)
        lifted = self.cube.pose.p[:, 2] > (TABLE_TOP_Z + CUBE_HALF + LIFT_SUCCESS)
        return {"success": is_grasped & lifted, "is_grasped": is_grasped, "lifted": lifted}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Simple shaped reward (not used by the scripted policy, but handy for RL).
        tcp_to_obj = torch.linalg.norm(self.cube.pose.p - self.agent.tcp_pose.p, axis=1)
        reward = 1 - torch.tanh(5 * tcp_to_obj)
        reward = reward + info["is_grasped"].float()
        reward = reward + 3.0 * info["success"].float()
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0


@register_env("AlohaMiniGripperView-v1", max_episode_steps=200)
class AlohaMiniGripperViewEnv(BaseEnv):
    """Empty scene with a close-up render camera on the LEFT gripper, used to
    visually + functionally verify the parallel-gripper integration (open/close,
    two symmetric pads). The left arm is posed to present the gripper to the camera."""

    SUPPORTED_ROBOTS = [ROBOT]
    VIEW_ARM = [0.0, 0.9, -0.9, 0.0, 0.0]   # left arm: presents the gripper to the cam
    # Gripper sits at ~[-0.194, -0.468, 0.924] in this pose; view it from +X.
    CAM_EYE = [0.12, -0.47, 1.00]
    CAM_TARGET = [-0.21, -0.47, 0.91]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids=ROBOT, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=self.CAM_EYE, target=self.CAM_TARGET)
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=self.CAM_EYE, target=self.CAM_TARGET)
        return CameraConfig("render_camera", pose, 640, 640, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.35, 0, 0]))

    def _load_scene(self, options: dict):
        # small backdrop so the gripper isn't rendered against pure black
        b = self.scene.create_actor_builder()
        b.add_box_visual(half_size=[0.01, 0.5, 0.5],
                         material=sapien.render.RenderMaterial(base_color=[0.2, 0.25, 0.3, 1]))
        b.initial_pose = sapien.Pose(p=[-0.5, -0.47, 0.9])
        b.build_static(name="backdrop")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            names = [j.name for j in self.agent.robot.active_joints]
            q = torch.zeros((len(env_idx), len(names)))
            arm = ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                   "left_wrist_flex", "left_wrist_roll"]
            for n, v in zip(arm, self.VIEW_ARM):
                q[:, names.index(n)] = v
            # start with the gripper open
            for n in ("left_finger_joint1", "left_finger_joint2"):
                q[:, names.index(n)] = 0.037
            self.agent.reset(q)
            self.agent.robot.set_pose(sapien.Pose(p=[-0.35, 0, 0]))

    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, dtype=bool, device=self.device)}

    def _get_obs_extra(self, info: dict):
        return dict()


@register_env("AlohaMiniPickCube-v1", max_episode_steps=100)
class AlohaMiniPickCubeEnv(PickCubeEnv):
    """Pick a cube and bring it to a goal position, single (left) arm."""

    SUPPORTED_ROBOTS = [ROBOT]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids=ROBOT, **kwargs)

    def _load_agent(self, options: dict):
        # Place the mobile base just behind the table edge. The virtual base /
        # lift joints then do the gross positioning at episode init.
        # Bypass PickCubeEnv's panda-specific spawn pose.
        from mani_skill.envs.sapien_env import BaseEnv
        BaseEnv._load_agent(self, options, sapien.Pose(p=[-0.35, 0, 0]))


@register_env("AlohaMiniPickPlace-v1", max_episode_steps=500)
class AlohaMiniPickPlaceEnv(AlohaMiniTablePickEnv):
    """Pick the red cube and place it on a nearby reachable goal platform."""

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.goal_marker = self._build_goal_platform("goal_platform", PICK_PLACE_GOAL_XY)

    def _build_goal_platform(self, name: str, xy: tuple[float, float]):
        b = self.scene.create_actor_builder()
        half = [PEDESTAL_HALF, PEDESTAL_HALF, TABLE_TOP_Z / 2]
        b.add_box_collision(half_size=half)
        b.add_box_visual(
            half_size=half,
            material=sapien.render.RenderMaterial(base_color=[0.2, 0.7, 0.35, 1]),
        )
        b.initial_pose = sapien.Pose(p=[xy[0], xy[1], TABLE_TOP_Z / 2])
        return b.build_static(name=name)

    def evaluate(self):
        goal_xy = torch.tensor(PICK_PLACE_GOAL_XY, device=self.device, dtype=torch.float32)
        cube_xy = self.cube.pose.p[:, :2]
        xy_close = torch.linalg.norm(cube_xy - goal_xy, axis=1) < 0.045
        z_target = TABLE_TOP_Z + CUBE_HALF
        z_close = torch.abs(self.cube.pose.p[:, 2] - z_target) < 0.08
        is_grasped = self.agent.is_grasping(self.cube, arm_id=1)
        return {
            "success": xy_close & z_close,
            "at_goal": xy_close,
            "height_ok": z_close,
            "is_grasped": is_grasped,
        }


@register_env("AlohaMiniStack-v1", max_episode_steps=550)
class AlohaMiniStackEnv(AlohaMiniTablePickEnv):
    """Pick the red cube and stack it on a second blue cube."""

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.stack_platform = self._build_stack_platform("stack_platform", STACK_BASE_XY)
        self.cube2 = actors.build_cube(
            self.scene,
            half_size=CUBE_HALF,
            color=[0.1, 0.2, 1.0, 1],
            name="cube2",
            initial_pose=sapien.Pose(
                p=[STACK_BASE_XY[0], STACK_BASE_XY[1], TABLE_TOP_Z + CUBE_HALF]
            ),
        )

    def _build_stack_platform(self, name: str, xy: tuple[float, float]):
        b = self.scene.create_actor_builder()
        half = [PEDESTAL_HALF, PEDESTAL_HALF, TABLE_TOP_Z / 2]
        b.add_box_collision(half_size=half)
        b.add_box_visual(
            half_size=half,
            material=sapien.render.RenderMaterial(base_color=[0.35, 0.35, 0.55, 1]),
        )
        b.initial_pose = sapien.Pose(p=[xy[0], xy[1], TABLE_TOP_Z / 2])
        return b.build_static(name=name)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = STACK_BASE_XY[0]
            xyz[:, 1] = STACK_BASE_XY[1]
            xyz[:, 2] = TABLE_TOP_Z + CUBE_HALF
            self.cube2.set_pose(Pose.create_from_pq(xyz))

    def evaluate(self):
        cube1_p = self.cube.pose.p
        cube2_p = self.cube2.pose.p
        xy_close = torch.linalg.norm(cube1_p[:, :2] - cube2_p[:, :2], axis=1) < 0.04
        stacked_z = cube1_p[:, 2] > cube2_p[:, 2] + CUBE_HALF * 1.5
        is_grasped = self.agent.is_grasping(self.cube, arm_id=1)
        return {
            "success": xy_close & stacked_z,
            "xy_close": xy_close,
            "stacked_z": stacked_z,
            "is_grasped": is_grasped,
        }


@register_env("AlohaMiniMultiYCB-v1", max_episode_steps=360)
class AlohaMiniMultiYCBEnv(BaseEnv):
    """Multi-object YCB tabletop pick task for AlohaMini data generation.

    All configured objects are present as distractors. The per-episode target
    cycles by episode_index and is exposed as ``env.unwrapped.target_object`` so
    the existing PickSkill can resolve it dynamically from config.
    """

    SUPPORTED_ROBOTS = [ROBOT, "aloha_mini_pro_v2", "aloha_mini_pro_v3"]

    def __init__(
        self,
        *args,
        object_ids: list[str] | None = None,
        object_xy_noise: float = 0.0,
        render_eye: list[float] | None = None,
        render_target: list[float] | None = None,
        robot_uid: str = ROBOT,
        base_xy: tuple[float, float] | None = None,
        slot_override_xy: list[tuple[float, float]] | None = None,
        **kwargs,
    ):
        self._robot_uid = robot_uid
        # The 6-DOF Pro arm has a smaller reach than the 5-DOF SO100, so its base can
        # be moved toward the objects. Default keeps the validated Std placement.
        self._base_xy = tuple(base_xy) if base_xy is not None else (-0.35, 0.0)
        # Optional per-index world-XY override for object placement (e.g. put the pick
        # object near the table edge so it is reachable from a base station OUTSIDE the
        # table footprint in the NAV/MANIP pick-and-place demo).
        self._slot_override_xy = (
            [tuple(p) for p in slot_override_xy] if slot_override_xy is not None else None
        )
        self.object_ids = list(object_ids or MULTI_YCB_DEFAULT_OBJECT_IDS)
        if not self.object_ids:
            raise ValueError("AlohaMiniMultiYCB-v1 requires at least one object id.")
        self.object_xy_noise = float(object_xy_noise)
        # Configurable human-render viewpoint. Default = side view that shows the whole
        # grasp; "head" replicates the cam_main mast camera (base_link + [0,0,1.45]).
        self._render_eye = list(render_eye) if render_eye is not None else [0.48, -0.88, 1.05]
        self._render_target = (
            list(render_target) if render_target is not None
            else [MULTI_TABLE_CENTER[0], MULTI_TABLE_CENTER[1], TABLE_TOP_Z + 0.05]
        )
        self.object_actor_names: list[str] = []
        self.object_actors: dict[str, Any] = {}
        self.object_asset_loaded: dict[str, bool] = {}
        self.object_z_offsets: dict[str, float] = {}
        self._episode_index = 0
        self._target_index = 0
        self.target_object_id = self.object_ids[0]
        self.target_actor_name = ""
        self.target_object: Any = None
        super().__init__(*args, robot_uids=self._robot_uid, **kwargs)

    @property
    def _default_sim_config(self):
        from mani_skill.utils.structs.types import SceneConfig, SimConfig

        return SimConfig(
            scene_config=SceneConfig(
                solver_position_iterations=30,
                solver_velocity_iterations=5,
                bounce_threshold=0.01,
            )
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=self._render_eye, target=self._render_target)
        return CameraConfig("render_camera", pose, 960, 640, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[self._base_xy[0], self._base_xy[1], 0]))

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.42, 0.40, 0.38])
        self.scene.add_directional_light([0.35, 0.45, -1.0], [2.4, 2.25, 2.05], shadow=True)
        self.scene.add_point_light([0.25, -0.85, 1.35], color=[2.2, 2.0, 1.75])
        self.scene.add_point_light([-0.55, -0.20, 1.15], color=[0.7, 0.8, 1.0])

    def _load_scene(self, options: dict):
        self._build_room_floor()
        self.table_top = self._build_box_static(
            "multi_ycb_table_top",
            half_size=[MULTI_TABLE_HALF[0], MULTI_TABLE_HALF[1], MULTI_TABLE_THICKNESS / 2],
            pose=[
                MULTI_TABLE_CENTER[0],
                MULTI_TABLE_CENTER[1],
                TABLE_TOP_Z - MULTI_TABLE_THICKNESS / 2,
            ],
            color=[0.50, 0.32, 0.18, 1.0],
        )
        self._build_table_legs()
        # NOTE: an earlier build disabled base<->table collision (bit 24) here so the Pro
        # base could sit inside the table footprint. With NAV/MANIP separation the base
        # now drives to a physically-valid station instead (feasibility-gated), so the
        # robot collides with the table normally — no more visual base/table overlap.

        self.object_actor_names = []
        self.object_actors = {}
        self.object_asset_loaded = {}
        for index, object_id in enumerate(self.object_ids):
            actor_name = _multi_ycb_actor_name(object_id)
            actor, asset_loaded = self._build_ycb_or_fallback(
                object_id=object_id,
                actor_name=actor_name,
                color=_fallback_color(index),
            )
            self.object_actor_names.append(actor_name)
            self.object_actors[object_id] = actor
            self.object_asset_loaded[object_id] = asset_loaded
            setattr(self, actor_name, actor)

        self.target_actor_name = self.object_actor_names[0]
        self.target_object = self.object_actors[self.target_object_id]

    def _after_reconfigure(self, options: dict):
        self.object_z_offsets = {}
        for object_id, actor in self.object_actors.items():
            fallback = MULTI_FALLBACK_CUBE_HALF
            self.object_z_offsets[object_id] = _actor_bottom_offset(actor, fallback)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        episode_index = int(options.get("episode_index", 0))
        self._episode_index = episode_index
        self._target_index = episode_index % len(self.object_ids)
        self.target_object_id = self.object_ids[self._target_index]
        self.target_actor_name = self.object_actor_names[self._target_index]
        self.target_object = self.object_actors[self.target_object_id]

        with torch.device(self.device):
            b = len(env_idx)
            self.agent.reset(self.agent.keyframes["ready"].qpos)
            # Spawn 9 mm up: the wheel meshes extend 7.3 mm below z=0, and with the
            # fixed root the resulting floor interpenetration produced ~10k contact
            # impulses that pinned the base (NAV commands had no effect at all).
            self.agent.robot.set_pose(sapien.Pose(p=[self._base_xy[0], self._base_xy[1], 0.009]))

            for index, object_id in enumerate(self.object_ids):
                actor = self.object_actors[object_id]
                if self._slot_override_xy is not None and index < len(self._slot_override_xy):
                    xy = np.asarray(self._slot_override_xy[index], dtype=np.float32)
                else:
                    xy = np.asarray(_multi_ycb_slot_xy(index), dtype=np.float32)
                if self.object_xy_noise > 0:
                    noise = (torch.rand((b, 2)) * 2 - 1) * self.object_xy_noise
                else:
                    noise = torch.zeros((b, 2))
                xyz = torch.zeros((b, 3))
                xyz[:, 0] = float(xy[0])
                xyz[:, 1] = float(xy[1])
                xyz[:, :2] += noise
                xyz[:, 2] = TABLE_TOP_Z + self.object_z_offsets.get(
                    object_id, MULTI_FALLBACK_CUBE_HALF
                )
                yaw = (episode_index * 0.73 + index * 1.19) % (2 * np.pi)
                quat = torch.tensor(
                    [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
                    dtype=torch.float32,
                ).repeat(b, 1)
                actor.set_pose(Pose.create_from_pq(xyz, quat))

            if not hasattr(self, "_target_rest_z"):
                self._target_rest_z = torch.zeros(self.num_envs, device=self.device)
            self._target_rest_z[env_idx] = TABLE_TOP_Z + self.object_z_offsets.get(
                self.target_object_id, MULTI_FALLBACK_CUBE_HALF
            )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            is_grasped=info["is_grasped"],
            target_pose=self.target_object.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_target=self.target_object.pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.target_object, arm_id=1)
        lifted = self.target_object.pose.p[:, 2] > (self._target_rest_z + LIFT_SUCCESS)
        return {
            "success": is_grasped & lifted,
            "is_grasped": is_grasped,
            "lifted": lifted,
            "target_object_id": self.target_object_id,
            "target_actor_name": self.target_actor_name,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj = torch.linalg.norm(
            self.target_object.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reward = 1 - torch.tanh(5 * tcp_to_obj)
        reward = reward + info["is_grasped"].float()
        reward = reward + 3.0 * info["success"].float()
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0

    def get_episode_metadata(self) -> dict[str, Any]:
        return {
            "episode_index": self._episode_index,
            "target_object_id": self.target_object_id,
            "target_actor_name": self.target_actor_name,
            "target_object_index": self._target_index,
            "table_top_z": TABLE_TOP_Z,
            "objects": [
                {
                    "object_id": object_id,
                    "actor_name": self.object_actor_names[index],
                    "asset_loaded": self.object_asset_loaded.get(object_id, False),
                    "slot_xy": list(_multi_ycb_slot_xy(index)),
                }
                for index, object_id in enumerate(self.object_ids)
            ],
        }

    def _build_room_floor(self):
        self.floor = self._build_box_static(
            "multi_ycb_floor",
            half_size=[0.95, 0.95, 0.01],
            pose=[-0.08, -0.35, -0.01],
            color=[0.56, 0.56, 0.54, 1.0],
        )

    def _build_table_legs(self):
        leg_half = [0.018, 0.018, (TABLE_TOP_Z - MULTI_TABLE_THICKNESS) / 2]
        leg_z = leg_half[2]
        for index, (sx, sy) in enumerate(
            [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ):
            x = MULTI_TABLE_CENTER[0] + sx * (MULTI_TABLE_HALF[0] - 0.035)
            y = MULTI_TABLE_CENTER[1] + sy * (MULTI_TABLE_HALF[1] - 0.035)
            self._build_box_static(
                f"multi_ycb_table_leg_{index}",
                half_size=leg_half,
                pose=[x, y, leg_z],
                color=[0.28, 0.20, 0.14, 1.0],
            )

    def _build_box_static(
        self,
        name: str,
        half_size: list[float],
        pose: list[float],
        color: list[float],
    ):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(
            half_size=half_size,
            material=sapien.render.RenderMaterial(base_color=color),
        )
        builder.initial_pose = sapien.Pose(p=pose)
        return builder.build_static(name=name)

    def _build_ycb_or_fallback(
        self,
        object_id: str,
        actor_name: str,
        color: list[float],
    ) -> tuple[Any, bool]:
        initial_pose = sapien.Pose()
        if _ycb_model_available(object_id):
            try:
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{object_id}")
                builder.initial_pose = initial_pose
                return builder.build(name=actor_name), True
            except Exception:
                pass
        return (
            actors.build_cube(
                self.scene,
                half_size=MULTI_FALLBACK_CUBE_HALF,
                color=color,
                name=actor_name,
                initial_pose=initial_pose,
            ),
            False,
        )


def _multi_ycb_actor_name(object_id: str) -> str:
    return "ycb_" + object_id.replace("-", "_").replace("/", "_")


def _multi_ycb_slot_xy(index: int) -> tuple[float, float]:
    offsets = [
        (0.0, 0.0),
        (0.075, 0.0),
        (-0.075, 0.0),
        (0.0, 0.075),
        (0.0, -0.075),
        (0.075, 0.075),
        (-0.075, 0.075),
        (0.075, -0.075),
        (-0.075, -0.075),
    ]
    if index < len(offsets):
        dx, dy = offsets[index]
    else:
        ring = 1 + index // len(offsets)
        angle = (index % len(offsets)) * (2 * np.pi / len(offsets))
        dx = 0.06 * ring * np.cos(angle)
        dy = 0.06 * ring * np.sin(angle)
    return MULTI_TABLE_CENTER[0] + dx, MULTI_TABLE_CENTER[1] + dy


def _fallback_color(index: int) -> list[float]:
    colors = [
        [0.90, 0.22, 0.18, 1.0],
        [0.14, 0.42, 0.85, 1.0],
        [0.22, 0.70, 0.35, 1.0],
        [0.93, 0.68, 0.20, 1.0],
        [0.55, 0.33, 0.74, 1.0],
    ]
    return colors[index % len(colors)]


def _load_ycb_info() -> dict[str, Any]:
    global _YCB_INFO_CACHE
    if _YCB_INFO_CACHE is None:
        info_path = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
        _YCB_INFO_CACHE = load_json(info_path) if info_path.exists() else {}
    return _YCB_INFO_CACHE


def _ycb_model_available(object_id: str) -> bool:
    model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / object_id
    return (
        object_id in _load_ycb_info()
        and (model_dir / "collision.ply").exists()
        and (model_dir / "textured.obj").exists()
    )


def _actor_bottom_offset(actor: Any, fallback: float) -> float:
    try:
        mesh = actor.get_first_collision_mesh()
        return float(-mesh.bounding_box.bounds[0, 2])
    except Exception:
        return float(fallback)

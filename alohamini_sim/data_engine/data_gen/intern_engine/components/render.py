"""Roll out a Sequence in ManiSkill and collect RGB observations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import RenderStageConfig
from ..registry import register_renderer
from ..types import Observations, Scene, Sequence


@register_renderer("mani_skill_renderer")
class ManiSkillRenderer:
    def __init__(self, cfg: RenderStageConfig) -> None:
        self.cfg = cfg
        self.camera_map = dict(cfg.camera_map)

    def render(self, scene: Scene, sequence: Sequence) -> Observations:
        env = scene.env
        obs = scene.reset_obs
        observations = Observations(
            images={self.camera_map.get(uid, uid): [] for uid in self.cfg.cameras},
            metadata={
                "camera_map": self.camera_map,
                "shader_pack": self.cfg.shader_pack,
            },
        )

        for frame_index, action_step in enumerate(sequence.steps):
            state = self._extract_state(env)
            action12 = self._action16_to_action12(action_step.action)
            observations.states.append(state)
            observations.actions.append(action12)
            observations.timestamps.append(frame_index / float(self.cfg.fps))
            self._collect_images(obs, observations)

            obs, reward, terminated, truncated, info = self._step(env, action_step.action)
            step_info = self._first_env_info(info)
            step_info["phase"] = action_step.phase
            step_info["duration"] = action_step.duration
            observations.step_infos.append(step_info)
            observations.rewards.append(float(self._first_scalar(reward, default=0.0)))
            observations.terminated.append(bool(self._first_scalar(terminated, default=False)))
            observations.truncated.append(bool(self._first_scalar(truncated, default=False)))
            if observations.terminated[-1] or observations.truncated[-1]:
                break
        self._tag_evaluation(env, sequence, observations)
        return observations

    def _collect_images(self, obs: Any, observations: Observations) -> None:
        sensor_data = obs.get("sensor_data", {}) if isinstance(obs, dict) else {}
        for cam_uid in self.cfg.cameras:
            feature_name = self.camera_map.get(cam_uid, cam_uid)
            cam_obs = sensor_data.get(cam_uid, {}) if isinstance(sensor_data, dict) else {}
            image = cam_obs.get("rgb") if isinstance(cam_obs, dict) else None
            if image is None:
                observations.missing_cameras[cam_uid] = (
                    observations.missing_cameras.get(cam_uid, 0) + 1
                )
                if self.cfg.missing_camera == "skip":
                    continue
                image_np = self._black_frame(cam_uid)
            else:
                image_np = self._to_uint8_hwc(image)
            observations.images.setdefault(feature_name, []).append(image_np)

    def _extract_state(self, env: Any) -> np.ndarray:
        robot = env.unwrapped.agent.robot
        qpos = self._to_numpy(robot.get_qpos())
        if qpos.ndim == 2:
            qpos = qpos[0]
        names = [joint.name for joint in robot.active_joints]
        index = {name: i for i, name in enumerate(names)}
        agent = env.unwrapped.agent
        left = self._named_values(qpos, index, getattr(agent, "left_arm_joint_names", []), 5)
        right = self._named_values(qpos, index, getattr(agent, "right_arm_joint_names", []), 5)
        left_grip = self._named_values(
            qpos, index, getattr(agent, "left_gripper_joint_names", []), 1
        )
        right_grip = self._named_values(
            qpos, index, getattr(agent, "right_gripper_joint_names", []), 1
        )
        # AlohaMini is natively 5-DOF per SO100 arm. Keep this parameterized so a
        # later 6-DOF Pro arm can extend the feature layout deliberately.
        return np.concatenate([left, left_grip, right, right_grip]).astype(np.float32)

    def _named_values(
        self, qpos: np.ndarray, index: dict[str, int], names: list[str], count: int
    ) -> np.ndarray:
        values: list[float] = []
        for name in names[:count]:
            if name in index:
                values.append(float(qpos[index[name]]))
        while len(values) < count:
            values.append(0.0)
        return np.asarray(values, dtype=np.float32)

    def _action16_to_action12(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 16:
            raise ValueError(f"Expected 16-D AlohaMini action, got {action.shape[0]}.")
        return np.concatenate([action[4:10], action[10:16]]).astype(np.float32)

    def _step(self, env: Any, action: np.ndarray) -> tuple[Any, Any, Any, Any, Any]:
        action_np = np.asarray(action, dtype=np.float32)
        try:
            result = env.step(action_np)
        except Exception:
            result = env.step(action_np[None, :])
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError(f"Unexpected env.step return: {type(result)!r}")
        return result

    def _to_uint8_hwc(self, value: Any) -> np.ndarray:
        arr = self._to_numpy(value)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected RGB image with 3 dims, got shape {arr.shape}.")
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype.kind == "f":
            arr = np.clip(arr * 255.0, 0, 255)
        return np.asarray(arr, dtype=np.uint8)

    def _black_frame(self, cam_uid: str) -> np.ndarray:
        shape = self.cfg.default_image_shapes.get(cam_uid, [128, 128, 3])
        return np.zeros(tuple(shape), dtype=np.uint8)

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _first_scalar(self, value: Any, default: Any) -> Any:
        if value is None:
            return default
        arr = self._to_numpy(value)
        if arr.shape == ():
            return arr.item()
        if arr.size == 0:
            return default
        return arr.reshape(-1)[0].item()

    def _first_env_info(self, info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        result: dict[str, Any] = {}
        for key, value in info.items():
            try:
                arr = self._to_numpy(value)
                if arr.shape == ():
                    result[key] = arr.item()
                elif arr.size:
                    first = arr.reshape(-1)[0]
                    result[key] = first.item() if hasattr(first, "item") else first
                else:
                    result[key] = value
            except Exception:
                result[key] = value
        return result

    def _tag_evaluation(
        self, env: Any, sequence: Sequence, observations: Observations
    ) -> None:
        evaluate = getattr(getattr(env, "unwrapped", env), "evaluate", None)
        if not callable(evaluate):
            return
        try:
            eval_info = self._first_env_info(evaluate())
        except Exception as exc:
            eval_info = {"error": repr(exc)}
        success = bool(eval_info.get("success", False))
        observations.metadata["evaluate"] = eval_info
        observations.metadata["success"] = success
        sequence.metadata["evaluate"] = eval_info
        sequence.metadata["success"] = success

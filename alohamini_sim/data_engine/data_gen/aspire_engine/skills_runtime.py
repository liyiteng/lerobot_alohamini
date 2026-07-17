"""Validated ASPIRE-style runtime skills for AlohaMini Pro.

This module is a refactor of the validated scratchpad programs:

* pro_nav_pick_place.py for station-gated pick, carry, and frozen-arm place
* pro_push_test.py for linear tabletop pushing

The motion recipes intentionally stay close to those scripts. The class only adds
episode tracing, shape validation, and command-facing method boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from data_gen.intern_engine.skills import build_skill
from data_gen.intern_engine.skills.ik import (
    actor_position,
    desired_approach_dir,
    resolve_actor,
    solve_arm_ik_full_pose,
)


V_BASE = 0.010
V_ARM = 0.020
V_ARM_DESCEND = 0.013
V_LIFT = 0.0045
CLOSE_STEPS = 32
SETTLE = 10
HOLD = 22
LIFT_HIGH = 0.16
PICK_BACK = 0.11
PITCH_DEFAULT = 60.0

TABLE_X = (-0.43, 0.17)
TABLE_Y = (-0.69, -0.21)
TABLE_Z = 0.70
PUSH_Z_OFFSET = 0.0
PUSH_START_BACK = 0.065
PUSH_END_OVERDRIVE = -0.070
PUSH_HOVER = 0.080
PUSH_WAYPOINT_SPACING = 0.010
PUSH_IK_ERR_LIMIT = 0.025


def interp(q0: np.ndarray | list[float], q1: np.ndarray | list[float], vmax: float) -> list[np.ndarray]:
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32)
    n = max(1, int(np.ceil(float(np.max(np.abs(q1 - q0))) / max(vmax, 1e-6))))
    return [q0 + (q1 - q0) * (k / n) for k in range(1, n + 1)]


def _best_full_pose(
    env: Any,
    target: np.ndarray,
    approach_dir: np.ndarray,
    jaw_dir: np.ndarray,
    arm: str,
    lift: float,
    seed: np.ndarray | None = None,
):
    """Full-pose IK seed sweep copied from the validated grasp demo."""

    best = None
    seeds = [seed] if seed is not None else [0.5, 1.0, 1.5, -0.5, 0.0]
    for s in seeds:
        kw: dict[str, Any] = dict(arm=arm, lift_position=lift, max_iters=250)
        if seed is not None:
            kw["seed"] = seed
        else:
            kw["shoulder_lift_seed"] = s
        r = solve_arm_ik_full_pose(env, target, approach_dir, jaw_dir, **kw)
        if best is None or r.error < best.error:
            best = r
    return best


def _cart_line(start: np.ndarray, end: np.ndarray, spacing: float) -> list[np.ndarray]:
    start = np.asarray(start, np.float32).reshape(3)
    end = np.asarray(end, np.float32).reshape(3)
    dist = float(np.linalg.norm(end - start))
    n = max(1, int(np.ceil(dist / max(spacing, 1e-6))))
    return [(start + (end - start) * (k / n)).astype(np.float32) for k in range(n + 1)]


def _as_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _vec3(value: Any) -> np.ndarray:
    a = _as_np(value)
    if a.ndim == 2:
        a = a[0]
    return np.asarray(a, dtype=np.float64).reshape(-1)[:3]


def _trace(ok: bool, info: dict[str, Any] | None = None, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": bool(ok), "info": info or {}, "evidence": evidence or {}}


@dataclass
class Station:
    station: np.ndarray
    arm_seed: np.ndarray
    target_pt: np.ndarray
    ik_error: float
    score: float


class SkillRuntime:
    """Runtime executor for validated ASPIRE atomic skills.

    `steps` contains only demonstration steps. Feasibility teleport-settle checks
    intentionally advance physics without recording, matching the source scripts.
    """

    def __init__(self, env: Any, record: bool = True) -> None:
        self.env = env
        self.be = env.unwrapped
        self.robot = self.be.agent.robot
        self.record = bool(record)
        self.steps: list[dict[str, Any]] = []
        self.traces: list[dict[str, Any]] = []

        names = [joint.name for joint in self.robot.active_joints]
        self.joint_names = names
        self.idx = {name: i for i, name in enumerate(names)}
        self.base_ids = [
            self.idx["root_x_axis_joint"],
            self.idx["root_y_axis_joint"],
            self.idx["root_z_rotation_joint"],
        ]

        root_p = self.robot.pose.p
        self.root_xy = _vec3(root_p)[:2].astype(np.float64)

        self.skill = build_skill("pick")
        self.lay = self.skill.arm_layout(self.be.agent)
        self.open_gripper = float(self.skill.open_gripper)
        self.closed_gripper = float(self.skill.closed_gripper)
        self.jaw_dir = np.array([1.0, 0.0, 0.0], np.float32)

        self.object_start: dict[str, np.ndarray] = {}
        for actor_name in getattr(self.be, "object_actor_names", []):
            self.object_start[actor_name] = actor_position(resolve_actor(self.env, actor_name)).copy()

        self.station_cache: dict[str, Station] = {}
        self.current_station: np.ndarray | None = None
        self.pick_station: np.ndarray | None = None
        self.grasp_q: np.ndarray | None = None
        self.holding_object: str | None = None
        self.lift_height = LIFT_HIGH

    def qnow(self) -> np.ndarray:
        q = self.robot.get_qpos()
        q = _as_np(q)
        if q.ndim == 2:
            q = q[0]
        return np.asarray(q, dtype=np.float64).reshape(-1).copy()

    def current_base_world(self) -> np.ndarray:
        base = self.qnow()[self.base_ids].astype(np.float32)
        base[0] += self.root_xy[0]
        base[1] += self.root_xy[1]
        return base

    def w2j(self, world_xy_yaw: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        # root_x/root_y/root_yaw joints are relative to robot.pose.p, not world XY.
        out = np.array(world_xy_yaw, np.float64).copy()
        out[0] -= self.root_xy[0]
        out[1] -= self.root_xy[1]
        return out

    def set_q(self, qpos: np.ndarray) -> None:
        self.robot.set_qpos(torch.as_tensor(np.asarray(qpos)[None], dtype=torch.float32))

    def arm_base_xy(self) -> np.ndarray:
        for link in self.be.agent.robot.get_links():
            if link.name == "left_base":
                return _vec3(link.pose.p)[:2].astype(np.float64)
        raise RuntimeError("left_base link not found")

    def left_base_world(self) -> np.ndarray:
        for link in self.be.agent.robot.get_links():
            if link.name == "left_base":
                return _vec3(link.pose.p).astype(np.float64)
        raise RuntimeError("left_base link not found")

    def object_name(self, object_name: str | None = None) -> str:
        if object_name is None:
            target = getattr(self.be, "target_actor_name", None)
            if target:
                return str(target)
            names = getattr(self.be, "object_actor_names", [])
            if names:
                return str(names[0])
            raise KeyError("environment has no object_actor_names")

        name = str(object_name)
        try:
            resolve_actor(self.env, name)
            return name
        except Exception:
            pass

        for actor_name in getattr(self.be, "object_actor_names", []):
            if name == actor_name or name in actor_name or actor_name.endswith(name):
                return str(actor_name)
        prefixed = f"ycb_{name}"
        try:
            resolve_actor(self.env, prefixed)
            return prefixed
        except Exception as exc:
            raise KeyError(f"could not resolve object {object_name!r}") from exc

    def _act(self, base_t: np.ndarray, arm_q: np.ndarray, grip: float, lift: float) -> np.ndarray:
        """base_t is WORLD (x, y) or (x, y, yaw); root joints are relative."""

        a = self.skill.current_action_template(self.env)
        yaw = float(base_t[2]) if len(base_t) > 2 else 0.0
        j = self.w2j((float(base_t[0]), float(base_t[1]), yaw))
        a[0], a[1], a[2] = float(j[0]), float(j[1]), float(j[2])
        a[3] = float(lift)
        a[self.lay["right_grip"]] = self.open_gripper
        self.skill.set_arm_action(a, "left", arm_q, grip)
        return a.astype(np.float32)

    def _step_unrecorded(self, action: np.ndarray) -> None:
        self.env.step(torch.as_tensor(action, dtype=torch.float32)[None])

    def _run(self, actions: list[np.ndarray], phase: str, extra: dict[str, Any] | None = None) -> None:
        for action in actions:
            obs, reward, terminated, truncated, info = self.env.step(
                torch.as_tensor(action, dtype=torch.float32)[None]
            )
            if self.record:
                self.steps.append(
                    {
                        "qpos": self.qnow().astype(np.float32),
                        "action": np.asarray(action, dtype=np.float32).reshape(-1),
                        "rgb": None,
                        "phase": phase,
                        "info": dict(extra or {}),
                    }
                )

    def _target_point(self, target_xy_or_xyz: np.ndarray, purpose: str) -> np.ndarray:
        arr = np.asarray(target_xy_or_xyz, dtype=np.float32).reshape(-1)
        if arr.size == 3:
            return arr.copy()
        if arr.size != 2:
            raise ValueError(f"{purpose} target must be xy or xyz, got shape {arr.shape}")
        if self.object_start:
            z = float(next(iter(self.object_start.values()))[2])
        else:
            z = 0.72
        return np.array([arr[0], arr[1], z], np.float32)

    def preselect_station(
        self,
        target_xy_or_xyz: np.ndarray,
        purpose: str,
        pitch_deg: float = PITCH_DEFAULT,
    ) -> dict[str, Any]:
        try:
            target_pt = self._target_point(target_xy_or_xyz, purpose)
            station = self._select_station(target_pt, purpose, pitch_deg)
            self.station_cache[purpose] = station
            trace = _trace(
                True,
                {
                    "skill": "preselect_station",
                    "purpose": purpose,
                    "station": station.station.tolist(),
                    "arm_seed": station.arm_seed.tolist(),
                },
                {
                    "target_pt": target_pt.tolist(),
                    "ik_error": station.ik_error,
                    "score": station.score,
                },
            )
        except Exception as exc:
            trace = _trace(False, {"skill": "preselect_station", "purpose": purpose}, {"error": str(exc)})
        self.traces.append(trace)
        return trace

    def _select_station(self, target_pt: np.ndarray, label: str, pitch_deg: float) -> Station:
        q0 = self.qnow()
        rest = self.skill.current_action_template(self.env)[self.lay["left_arm"]].astype(np.float32)
        best: tuple[float, float, float, float, np.ndarray, float] | None = None

        # yaw candidates and offsets are copied from the validated source.
        armoff = {0.0: (0.156, -0.041), -np.pi / 2: (-0.041, -0.156)}
        for yaw, (ox, oy) in armoff.items():
            for dx in (-0.10, -0.05, 0.0, 0.05, 0.10):
                for by in (0.08, 0.04, 0.01):
                    bx = float(target_pt[0] - ox + dx)
                    j = self.w2j((bx, by, yaw))
                    q = q0.copy()
                    q[self.base_ids[0]], q[self.base_ids[1]], q[self.base_ids[2]] = j[0], j[1], j[2]
                    self.set_q(q)

                    hold = self._act(np.array([bx, by, yaw], np.float32), rest, self.open_gripper, 0.0)
                    for _ in range(6):
                        self._step_unrecorded(hold)
                    qs = self.qnow()
                    base_err = float(np.hypot(qs[self.base_ids[0]] - j[0], qs[self.base_ids[1]] - j[1]))
                    if base_err > 0.02:
                        continue

                    ab = self.arm_base_xy()
                    appr_dir = desired_approach_dir(target_pt, pitch_deg, base_xy=tuple(ab)).astype(np.float32)
                    r = solve_arm_ik_full_pose(
                        self.env,
                        target_pt,
                        appr_dir,
                        self.jaw_dir,
                        arm="left",
                        lift_position=0.0,
                        shoulder_lift_seed=1.0,
                        max_iters=120,
                    )
                    dist = float(np.linalg.norm(ab - target_pt[:2]))
                    comfort = abs(dist - 0.20)
                    score = r.error + 0.02 * comfort
                    if best is None or score < best[0]:
                        best = (score, bx, by, yaw, r.arm_qpos.copy(), float(r.error))

        self.set_q(q0)
        if best is None:
            raise RuntimeError(f"no physically-valid station found for {label}")
        return Station(
            station=np.array([best[1], best[2], best[3]], np.float32),
            arm_seed=best[4].astype(np.float32),
            target_pt=np.asarray(target_pt, np.float32).copy(),
            ik_error=float(best[5]),
            score=float(best[0]),
        )

    def goto_station(self, target_xy: np.ndarray, purpose: str) -> dict[str, Any]:
        """Feasibility-gated station selection plus base navigation."""

        try:
            station_obj = self.station_cache.get(purpose)
            if station_obj is None:
                station_obj = self._select_station(self._target_point(target_xy, purpose), purpose, PITCH_DEFAULT)
                self.station_cache[purpose] = station_obj

            rest_q = self.skill.current_action_template(self.env)[self.lay["left_arm"]].astype(np.float32)
            base0 = self.current_base_world()
            nav = [self._act(b, rest_q, self.open_gripper, 0.0) for b in interp(base0, station_obj.station, V_BASE)]
            if nav:
                nav += [nav[-1]] * 20
            self._run(nav, f"goto_station/{purpose}", {"purpose": purpose})
            self.current_station = station_obj.station.copy()
            if purpose == "pick":
                self.pick_station = station_obj.station.copy()
            trace = _trace(
                True,
                {
                    "skill": "goto_station",
                    "purpose": purpose,
                    "station": station_obj.station.tolist(),
                    "arm_seed": station_obj.arm_seed.tolist(),
                },
                {
                    "target_pt": station_obj.target_pt.tolist(),
                    "ik_error": station_obj.ik_error,
                    "steps": len(nav),
                },
            )
        except Exception as exc:
            trace = _trace(False, {"skill": "goto_station", "purpose": purpose}, {"error": str(exc)})
        self.traces.append(trace)
        return trace

    def pick(self, object_name: str, pitch_deg: float = PITCH_DEFAULT) -> dict[str, Any]:
        try:
            name = self.object_name(object_name)
            obj_actor = resolve_actor(self.env, name)
            obj0 = self.object_start.get(name, actor_position(obj_actor).copy())
            if "pick" not in self.station_cache:
                self.goto_station(actor_position(obj_actor), "pick")
                if "pick" not in self.station_cache:
                    raise RuntimeError("pick station unavailable")
            st1 = self.station_cache["pick"].station
            st1_arm_seed = self.station_cache["pick"].arm_seed

            grasp_pt = actor_position(obj_actor).astype(np.float32)
            lbw = self.left_base_world()
            appr_dir = desired_approach_dir(grasp_pt, pitch_deg, base_xy=tuple(lbw[:2])).astype(np.float32)
            pre_pt = (grasp_pt - appr_dir * PICK_BACK).astype(np.float32)

            # Validated desc-first IK branch: solve grasp pose first, then pre-grasp.
            desc = _best_full_pose(self.env, grasp_pt, appr_dir, self.jaw_dir, "left", 0.0, seed=st1_arm_seed)
            appr = _best_full_pose(self.env, pre_pt, appr_dir, self.jaw_dir, "left", 0.0, seed=desc.arm_qpos)

            descent = [appr.arm_qpos]
            seedq = appr.arm_qpos
            for s in np.linspace(PICK_BACK, 0.0, 9)[1:]:
                w = _best_full_pose(
                    self.env,
                    (grasp_pt - appr_dir * s).astype(np.float32),
                    appr_dir,
                    self.jaw_dir,
                    "left",
                    0.0,
                    seed=seedq,
                )
                descent.append(w.arm_qpos)
                seedq = w.arm_qpos
            desc_q = descent[-1]

            rest_q = self.skill.current_action_template(self.env)[self.lay["left_arm"]].astype(np.float32)
            actions: list[np.ndarray] = []
            for q in interp(rest_q, appr.arm_qpos, V_ARM):
                actions.append(self._act(st1, q, self.open_gripper, 0.0))
            for q0, q1 in zip(descent[:-1], descent[1:]):
                for q in interp(q0, q1, V_ARM_DESCEND):
                    actions.append(self._act(st1, q, self.open_gripper, 0.0))
            for k in range(1, CLOSE_STEPS + 1):
                grip = self.open_gripper + (self.closed_gripper - self.open_gripper) * k / CLOSE_STEPS
                actions.append(self._act(st1, desc_q, grip, 0.0))
            for _ in range(SETTLE):
                actions.append(self._act(st1, desc_q, self.closed_gripper, 0.0))
            for lz in interp([0.0], [LIFT_HIGH], V_LIFT):
                actions.append(self._act(st1, desc_q, self.closed_gripper, float(lz[0])))
            self._run(actions, "pick", {"object": name})

            held = actor_position(obj_actor).copy()
            ok = bool(held[2] > obj0[2] + 0.05)
            if ok:
                self.holding_object = name
                self.grasp_q = desc_q.astype(np.float32)
                self.current_station = st1.copy()
            trace = _trace(
                ok,
                {"skill": "pick", "object": name, "pitch_deg": float(pitch_deg)},
                {
                    "object_start": obj0.tolist(),
                    "object_after": held.tolist(),
                    "lift_delta": float(held[2] - obj0[2]),
                    "ik_approach_error": float(appr.error),
                    "ik_desc_error": float(desc.error),
                    "cartesian_waypoints": 9,
                    "steps": len(actions),
                },
            )
        except Exception as exc:
            trace = _trace(False, {"skill": "pick", "object": str(object_name)}, {"error": str(exc)})
        self.traces.append(trace)
        return trace

    def carry(self, to_station: np.ndarray | str | dict[str, Any]) -> dict[str, Any]:
        try:
            if self.holding_object is None or self.grasp_q is None:
                raise RuntimeError("carry requires a successful pick first")
            if isinstance(to_station, str):
                station = self.station_cache[to_station].station
            elif isinstance(to_station, dict) and "station" in to_station:
                station = np.asarray(to_station["station"], np.float32)
            else:
                station = np.asarray(to_station, np.float32).reshape(3)
            start = self.current_station.copy() if self.current_station is not None else self.current_base_world()

            actions = [
                self._act(b, self.grasp_q, self.closed_gripper, LIFT_HIGH)
                for b in interp(start, station, V_BASE * 0.5)
            ]
            if actions:
                actions += [actions[-1]] * 20
            self._run(actions, "carry", {"object": self.holding_object})
            self.current_station = station.copy()

            obj_actor = resolve_actor(self.env, self.holding_object)
            obj0 = self.object_start[self.holding_object]
            carried = actor_position(obj_actor).copy()
            ok = bool(carried[2] > obj0[2] + 0.05)

            t1p = _vec3(self.be.agent.left_finger1_tip.pose.p)
            t2p = _vec3(self.be.agent.left_finger2_tip.pose.p)
            grasp_off = (carried[:2] - (t1p + t2p)[:2] / 2).astype(np.float32)
            trace = _trace(
                ok,
                {"skill": "carry", "object": self.holding_object, "station": station.tolist()},
                {
                    "object_after": carried.tolist(),
                    "object_start": obj0.tolist(),
                    "grasp_offset_xy": grasp_off.tolist(),
                    "steps": len(actions),
                },
            )
        except Exception as exc:
            trace = _trace(False, {"skill": "carry"}, {"error": str(exc)})
        self.traces.append(trace)
        return trace

    def place(self, target_xy: np.ndarray) -> dict[str, Any]:
        try:
            if self.holding_object is None or self.grasp_q is None:
                raise RuntimeError("place requires a successful pick first")
            name = self.holding_object
            obj_actor = resolve_actor(self.env, name)
            obj0 = self.object_start[name]
            target_xy = np.asarray(target_xy, dtype=np.float32).reshape(2)
            cur_base = self.current_station.copy() if self.current_station is not None else self.current_base_world()

            align_steps = 0
            align_rounds = 0
            for _ in range(3):
                obj_now = actor_position(obj_actor)
                delta = np.array([target_xy[0] - obj_now[0], target_xy[1] - obj_now[1]], np.float32)
                if float(np.linalg.norm(delta)) < 0.008:
                    break
                tgt = cur_base.copy()
                tgt[0] += delta[0]
                tgt[1] += delta[1]
                actions = [
                    self._act(b, self.grasp_q, self.closed_gripper, LIFT_HIGH)
                    for b in interp(cur_base, tgt, V_BASE * 0.5)
                ]
                actions += [self._act(tgt, self.grasp_q, self.closed_gripper, LIFT_HIGH)] * 10
                self._run(actions, "place/base_align", {"object": name})
                align_steps += len(actions)
                align_rounds += 1
                cur_base = tgt

            obj_now = actor_position(obj_actor)
            drop = float(obj_now[2] - (obj0[2] + 0.02))
            lift_lo = max(LIFT_HIGH - drop, -0.1)
            actions = [
                self._act(cur_base, self.grasp_q, self.closed_gripper, float(lz[0]))
                for lz in interp([LIFT_HIGH], [lift_lo], V_LIFT)
            ]
            actions += [self._act(cur_base, self.grasp_q, self.closed_gripper, lift_lo)] * 15
            self._run(actions, "place/lift_descent", {"object": name})
            pre_release = actor_position(obj_actor).copy()

            release_actions: list[np.ndarray] = []
            for k in range(1, 11):
                grip = self.closed_gripper + (self.open_gripper - self.closed_gripper) * k / 10
                release_actions.append(self._act(cur_base, self.grasp_q, grip, lift_lo))
            for _ in range(10):
                release_actions.append(self._act(cur_base, self.grasp_q, self.open_gripper, lift_lo))
            for lz in interp([lift_lo], [LIFT_HIGH], V_LIFT):
                release_actions.append(self._act(cur_base, self.grasp_q, self.open_gripper, float(lz[0])))
            for _ in range(HOLD):
                release_actions.append(self._act(cur_base, self.grasp_q, self.open_gripper, LIFT_HIGH))
            self._run(release_actions, "place/release_retreat", {"object": name})

            objf = actor_position(obj_actor).copy()
            err = float(np.linalg.norm(objf[:2] - target_xy))
            ok = bool(err < 0.06)
            self.current_station = cur_base.copy()
            if ok:
                self.holding_object = None
            trace = _trace(
                ok,
                {"skill": "place", "object": name, "target_xy": target_xy.tolist()},
                {
                    "object_start": obj0.tolist(),
                    "pre_release": pre_release.tolist(),
                    "object_final": objf.tolist(),
                    "xy_error": err,
                    "z_error": float(abs(objf[2] - obj0[2])),
                    "lift_low": float(lift_lo),
                    "align_rounds": align_rounds,
                    "align_steps": align_steps,
                    "release_steps": len(release_actions),
                },
            )
        except Exception as exc:
            trace = _trace(False, {"skill": "place"}, {"error": str(exc)})
        self.traces.append(trace)
        return trace

    def push(self, object_name: str, target_xy: np.ndarray) -> dict[str, Any]:
        """Linear push ported from pro_push_test.py."""

        try:
            name = self.object_name(object_name)
            obj_actor = resolve_actor(self.env, name)
            obj0 = actor_position(obj_actor).copy()
            target_xy = np.asarray(target_xy, np.float32).reshape(2)
            dir_xy = target_xy - obj0[:2]
            norm = float(np.linalg.norm(dir_xy))
            if norm < 1e-6:
                raise RuntimeError("push target is already at object xy")
            dir_xy = (dir_xy / norm).astype(np.float32)
            push_dir = np.array([dir_xy[0], dir_xy[1], 0.0], np.float32)
            push_z = float(obj0[2] + PUSH_Z_OFFSET)

            start_pt = np.array([obj0[0], obj0[1], push_z], np.float32) - push_dir * PUSH_START_BACK
            end_pt = np.array([target_xy[0], target_xy[1], push_z], np.float32) + push_dir * PUSH_END_OVERDRIVE
            hover_pt = start_pt + np.array([0.0, 0.0, PUSH_HOVER], np.float32)

            station, station_seed = self._select_push_station([start_pt, end_pt], push_dir)
            rest_q = self.skill.current_action_template(self.env)[self.lay["left_arm"]].astype(np.float32)
            base0 = self.current_base_world()
            nav = [self._act(b, rest_q, self.closed_gripper, 0.0) for b in interp(base0, station, V_BASE)]
            if nav:
                nav += [nav[-1]] * 20
            self._run(nav, "push/nav", {"object": name})

            obj_pre = actor_position(obj_actor).copy()
            start_pt = np.array([obj_pre[0], obj_pre[1], push_z], np.float32) - push_dir * PUSH_START_BACK
            end_pt = np.array([target_xy[0], target_xy[1], push_z], np.float32) + push_dir * PUSH_END_OVERDRIVE
            hover_pt = start_pt + np.array([0.0, 0.0, PUSH_HOVER], np.float32)

            hover_q = self._solve_push_path([hover_pt], push_dir, station_seed, "hover")[0]
            descend_points = _cart_line(hover_pt, start_pt, PUSH_WAYPOINT_SPACING)[1:]
            descend_qs = self._solve_push_path(descend_points, push_dir, hover_q, "behind_descend")
            push_points = _cart_line(start_pt, end_pt, PUSH_WAYPOINT_SPACING)[1:]
            push_qs = self._solve_push_path(push_points, push_dir, descend_qs[-1], "push_line")

            actions: list[np.ndarray] = []
            for q in interp(rest_q, hover_q, V_ARM):
                actions.append(self._act(station, q, self.closed_gripper, 0.0))
            prev = hover_q
            for q_next in descend_qs:
                for q in interp(prev, q_next, V_ARM_DESCEND):
                    actions.append(self._act(station, q, self.closed_gripper, 0.0))
                prev = q_next
            self._run(actions, "push/approach", {"object": name})

            push_actions: list[np.ndarray] = []
            prev = descend_qs[-1]
            for q_next in push_qs:
                for q in interp(prev, q_next, V_ARM_DESCEND):
                    push_actions.append(self._act(station, q, self.closed_gripper, 0.0))
                prev = q_next
            self._run(push_actions, "push/line", {"object": name})

            hold = [self._act(station, push_qs[-1], self.closed_gripper, 0.0)] * 20
            self._run(hold, "push/settle", {"object": name})

            objf = actor_position(obj_actor).copy()
            xy_err = float(np.linalg.norm(objf[:2] - target_xy))
            on_table = (
                TABLE_X[0] <= float(objf[0]) <= TABLE_X[1]
                and TABLE_Y[0] <= float(objf[1]) <= TABLE_Y[1]
                and float(objf[2]) >= TABLE_Z - 0.005
            )
            ok = bool(xy_err <= 0.05 and on_table)
            trace = _trace(
                ok,
                {"skill": "push", "object": name, "target_xy": target_xy.tolist()},
                {
                    "object_start": obj0.tolist(),
                    "object_final": objf.tolist(),
                    "xy_error": xy_err,
                    "on_table": bool(on_table),
                    "station": station.tolist(),
                    "push_dir": push_dir.tolist(),
                },
            )
        except Exception as exc:
            trace = _trace(False, {"skill": "push", "object": str(object_name)}, {"error": str(exc)})
        self.traces.append(trace)
        return trace

    def _solve_push_path(
        self,
        points: list[np.ndarray],
        approach_dir: np.ndarray,
        seed: np.ndarray | None,
        label: str,
    ) -> list[np.ndarray]:
        qs: list[np.ndarray] = []
        errs: list[float] = []
        seed_q = seed
        for pt in points:
            if seed_q is None:
                r = _best_full_pose(self.env, pt, approach_dir, self.jaw_dir, "left", 0.0)
            else:
                r = solve_arm_ik_full_pose(
                    self.env,
                    pt,
                    approach_dir,
                    self.jaw_dir,
                    arm="left",
                    lift_position=0.0,
                    seed=seed_q,
                    max_iters=250,
                )
                if r.error > 0.015:
                    alt = _best_full_pose(self.env, pt, approach_dir, self.jaw_dir, "left", 0.0)
                    if alt.error < r.error:
                        r = alt
            qs.append(r.arm_qpos.copy())
            errs.append(float(r.error))
            seed_q = r.arm_qpos
        if errs and max(errs) > PUSH_IK_ERR_LIMIT:
            raise RuntimeError(f"{label} IK max error {max(errs):.4f} exceeds {PUSH_IK_ERR_LIMIT:.4f}")
        return qs

    def _select_push_station(
        self,
        target_points: list[np.ndarray],
        approach_dir: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q0 = self.qnow()
        rest = self.skill.current_action_template(self.env)[self.lay["left_arm"]].astype(np.float32)
        best: tuple[float, float, float, float, np.ndarray, list[float], np.ndarray, float] | None = None
        armoff = {-np.pi / 2: (-0.041, -0.156)}
        for yaw, (ox, oy) in armoff.items():
            for dx in (-0.04, 0.0, 0.04):
                for by in (0.04, 0.01):
                    bx = float(target_points[0][0] - ox + dx)
                    j = self.w2j((bx, by, yaw))
                    q = q0.copy()
                    q[self.base_ids[0]], q[self.base_ids[1]], q[self.base_ids[2]] = j[0], j[1], j[2]
                    self.set_q(q)
                    hold = self._act(np.array([bx, by, yaw], np.float32), rest, self.open_gripper, 0.0)
                    for _ in range(6):
                        self._step_unrecorded(hold)
                    qs_now = self.qnow()
                    base_err = float(np.hypot(qs_now[self.base_ids[0]] - j[0], qs_now[self.base_ids[1]] - j[1]))
                    if base_err > 0.02:
                        continue

                    ab = self.arm_base_xy()
                    seed_q = None
                    errs: list[float] = []
                    qs_path: list[np.ndarray] = []
                    for pt in target_points:
                        kw: dict[str, Any] = dict(arm="left", lift_position=0.0, max_iters=120)
                        if seed_q is None:
                            kw["shoulder_lift_seed"] = 1.0
                        else:
                            kw["seed"] = seed_q
                        r = solve_arm_ik_full_pose(self.env, pt, approach_dir, self.jaw_dir, **kw)
                        errs.append(float(r.error))
                        qs_path.append(r.arm_qpos.copy())
                        seed_q = r.arm_qpos
                    max_err = max(errs)
                    if max_err > PUSH_IK_ERR_LIMIT:
                        continue
                    dist = float(np.linalg.norm(ab - target_points[0][:2]))
                    comfort = abs(dist - 0.20)
                    score = max_err + 0.02 * comfort
                    if best is None or score < best[0]:
                        best = (score, bx, by, yaw, qs_path[0], errs, ab.copy(), base_err)
        self.set_q(q0)
        if best is None:
            raise RuntimeError("no physically-valid push station found")
        return np.array([best[1], best[2], best[3]], np.float32), best[4].astype(np.float32)

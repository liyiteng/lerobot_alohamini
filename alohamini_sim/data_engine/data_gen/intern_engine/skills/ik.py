"""Reusable 5-DOF AlohaMini IK helpers for skill planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class IKResult:
    arm: str
    target: np.ndarray
    qpos: np.ndarray
    arm_qpos: np.ndarray
    success: bool
    error: float
    iterations: int
    wrist_roll: float | None = None
    wrist_roll_score: float | None = None


def solve_arm_ik_position(
    env: Any,
    target: np.ndarray,
    *,
    arm: str = "left",
    seed: np.ndarray | None = None,
    lift_position: float | None = None,
    shoulder_lift_seed: float | None = 1.0,
    max_iters: int = 80,
    tol: float = 0.005,
    damping: float = 0.05,
    fd_eps: float = 1e-3,
    max_joint_step: float = 0.12,
) -> IKResult:
    """Damped-least-squares position-only IK over one native 5-DOF SO100 arm.

    A future 6-DOF AlohaMini Pro arm should extend this helper by adding an
    orientation residual and sixth arm joint, rather than overloading the
    current 5-DOF position-only solve.
    """

    target = np.asarray(target, dtype=np.float64).reshape(3)
    env_unwrapped = getattr(env, "unwrapped", env)
    agent = env_unwrapped.agent
    robot = agent.robot
    qpos0 = get_active_qpos(robot)
    names = [joint.name for joint in robot.active_joints]
    index = {name: i for i, name in enumerate(names)}
    arm_names = _arm_joint_names(agent, arm)
    arm_indices = [index[name] for name in arm_names]
    lo, hi = _arm_joint_limits(robot, arm_indices)

    q_work = qpos0.copy()
    if lift_position is not None:
        for name in getattr(agent, "lift_joint_names", []):
            if name in index:
                q_work[index[name]] = float(lift_position)

    if seed is None:
        q_arm = q_work[arm_indices].astype(np.float64)
        if shoulder_lift_seed is not None and q_arm.size > 1:
            q_arm[1] = float(shoulder_lift_seed)
    else:
        q_arm = np.asarray(seed, dtype=np.float64).reshape(len(arm_indices))
    q_arm = np.clip(q_arm, lo, hi)

    best_qpos = q_work.copy()
    best_qpos[arm_indices] = q_arm
    best_error = float("inf")
    success = False
    iterations = 0

    with temporary_robot_qpos(robot):
        for iterations in range(1, max_iters + 1):
            q_work[arm_indices] = q_arm
            set_active_qpos(robot, q_work)
            pos = tcp_position(agent, arm)
            err = target - pos
            err_norm = float(np.linalg.norm(err))
            if err_norm < best_error:
                best_error = err_norm
                best_qpos = q_work.copy()
            if err_norm <= tol:
                success = True
                break

            jac = np.zeros((3, len(arm_indices)), dtype=np.float64)
            for col in range(len(arm_indices)):
                q_probe = q_arm.copy()
                q_probe[col] += fd_eps
                q_work[arm_indices] = q_probe
                set_active_qpos(robot, q_work)
                jac[:, col] = (tcp_position(agent, arm) - pos) / fd_eps

            lhs = jac @ jac.T + (damping * damping) * np.eye(3)
            try:
                dq = jac.T @ np.linalg.solve(lhs, err)
            except np.linalg.LinAlgError:
                dq = jac.T @ np.linalg.pinv(lhs) @ err
            dq = np.clip(dq, -max_joint_step, max_joint_step)
            q_arm = np.clip(q_arm + dq, lo, hi)

    best_arm_qpos = best_qpos[arm_indices].astype(np.float32)
    return IKResult(
        arm=arm,
        target=target.astype(np.float32),
        qpos=best_qpos.astype(np.float32),
        arm_qpos=best_arm_qpos,
        success=success,
        error=best_error,
        iterations=iterations,
    )


def palm_position(agent: Any, arm: str) -> np.ndarray:
    """World position of the gripper palm (Fixed_Jaw) link origin."""
    link = getattr(agent, "left_palm_link" if arm == "left" else "right_palm_link")
    p = to_numpy(link.pose.p)
    if p.ndim == 2:
        p = p[0]
    return np.asarray(p, dtype=np.float64).reshape(-1)[:3]


def gripper_approach_axis(agent: Any, arm: str) -> np.ndarray:
    """Unit world vector the gripper points along (palm origin -> finger-tip midpoint).

    Derived purely from link geometry, so it needs no orientation-convention
    calibration: at a top-down grasp the tips sit below the palm, giving ~(0,0,-1).
    """
    d = tcp_position(agent, arm) - palm_position(agent, arm)
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])


def desired_approach_dir(target: np.ndarray, pitch_deg: float,
                         base_xy: tuple[float, float] = (-0.35, 0.0)) -> np.ndarray:
    """Approach unit vector for a given pitch (90=straight down, 0=horizontal),
    leaning horizontally from the arm base toward the target."""
    horiz = np.array([float(target[0]) - base_xy[0], float(target[1]) - base_xy[1]], np.float64)
    hn = float(np.linalg.norm(horiz))
    horiz = horiz / hn if hn > 1e-6 else np.array([0.0, -1.0])
    pr = np.radians(float(pitch_deg))
    return np.array([np.cos(pr) * horiz[0], np.cos(pr) * horiz[1], -np.sin(pr)], np.float64)


def solve_arm_ik_pose(
    env: Any,
    target: np.ndarray,
    approach_pitch_deg: float,
    *,
    arm: str = "left",
    seed: np.ndarray | None = None,
    lift_position: float | None = None,
    shoulder_lift_seed: float | None = 1.0,
    base_xy: tuple[float, float] = (-0.35, 0.0),
    dir_weight: float = 0.10,
    max_iters: int = 140,
    tol_pos: float = 0.005,
    tol_dir: float = 0.06,
    damping: float = 0.05,
    fd_eps: float = 1e-3,
    max_joint_step: float = 0.12,
) -> IKResult:
    """Position + approach-pitch IK for the 5-DOF SO100 arm.

    Solves arm joints 0..3 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex) for
    BOTH the TCP position and the gripper approach direction; wrist_roll (joint 4) is
    held at its seed value (it rotates about the approach axis, so it does not affect
    position or approach pitch -- run apply_horizontal_jaw_wrist_roll afterwards to set
    it for horizontal jaws). Lets grasps approach top-down (pitch=90) or tilted.
    """
    target = np.asarray(target, dtype=np.float64).reshape(3)
    env_unwrapped = getattr(env, "unwrapped", env)
    agent = env_unwrapped.agent
    robot = agent.robot
    qpos0 = get_active_qpos(robot)
    names = [joint.name for joint in robot.active_joints]
    index = {name: i for i, name in enumerate(names)}
    arm_names = _arm_joint_names(agent, arm)
    arm_indices = [index[name] for name in arm_names]
    solve_cols = arm_indices[:-1]  # keep the last arm joint (tool roll) fixed
    n_solve = len(solve_cols)

    q_work = qpos0.copy()
    if lift_position is not None:
        for name in getattr(agent, "lift_joint_names", []):
            if name in index:
                q_work[index[name]] = float(lift_position)

    if seed is None:
        q_arm = q_work[arm_indices].astype(np.float64)
        if shoulder_lift_seed is not None and q_arm.size > 1:
            q_arm[1] = float(shoulder_lift_seed)
    else:
        q_arm = np.asarray(seed, dtype=np.float64).reshape(len(arm_indices))
    q_work[arm_indices] = q_arm

    d_des = desired_approach_dir(target, approach_pitch_deg, base_xy)

    best_qpos = q_work.copy()
    best_score = float("inf")
    best_pos = float("inf")
    best_dir = float("inf")
    success = False
    iterations = 0

    def residual() -> tuple[np.ndarray, float, float]:
        pos = tcp_position(agent, arm)
        d_cur = gripper_approach_axis(agent, arm)
        e_pos = target - pos
        e_dir = d_des - d_cur
        r = np.concatenate([e_pos, dir_weight * e_dir])
        return r, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_dir))

    with temporary_robot_qpos(robot):
        for iterations in range(1, max_iters + 1):
            q_work[solve_cols] = q_arm[:n_solve]
            set_active_qpos(robot, q_work)
            r, pos_err, dir_err = residual()
            score = pos_err + 0.3 * dir_err
            if score < best_score:
                best_score, best_pos, best_dir = score, pos_err, dir_err
                best_qpos = q_work.copy()
            if pos_err <= tol_pos and dir_err <= tol_dir:
                success = True
                break

            jac = np.zeros((6, n_solve), dtype=np.float64)
            for c in range(n_solve):
                q_probe = q_arm[:n_solve].copy()
                q_probe[c] += fd_eps
                q_work[solve_cols] = q_probe
                set_active_qpos(robot, q_work)
                r_p, _, _ = residual()
                jac[:, c] = (r_p - r) / fd_eps * -1.0  # d(residual)/dq; residual=goal-current
            # solve J dq = r  (move current toward goal)
            lhs = jac @ jac.T + (damping * damping) * np.eye(6)
            try:
                dq = jac.T @ np.linalg.solve(lhs, r)
            except np.linalg.LinAlgError:
                dq = jac.T @ np.linalg.pinv(lhs) @ r
            dq = np.clip(dq, -max_joint_step, max_joint_step)
            q_arm[:n_solve] = q_arm[:n_solve] + dq
            q_work[solve_cols] = q_arm[:n_solve]

    best_arm_qpos = best_qpos[arm_indices].astype(np.float32)
    return IKResult(
        arm=arm,
        target=target.astype(np.float32),
        qpos=best_qpos.astype(np.float32),
        arm_qpos=best_arm_qpos,
        success=success,
        error=best_pos,
        iterations=iterations,
        wrist_roll=None,
        wrist_roll_score=best_dir,
    )


def jaw_axis(agent: Any, arm: str) -> np.ndarray:
    """Unit world vector along the finger-tip separation (jaw opening direction)."""
    prefix = "left" if arm == "left" else "right"
    p1 = to_numpy(getattr(agent, f"{prefix}_finger1_tip").pose.p)
    p2 = to_numpy(getattr(agent, f"{prefix}_finger2_tip").pose.p)
    if p1.ndim == 2:
        p1 = p1[0]
    if p2.ndim == 2:
        p2 = p2[0]
    d = np.asarray(p2, np.float64).reshape(-1)[:3] - np.asarray(p1, np.float64).reshape(-1)[:3]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def solve_arm_ik_full_pose(
    env: Any,
    target: np.ndarray,
    approach_dir: np.ndarray,
    jaw_dir: np.ndarray,
    *,
    arm: str = "left",
    seed: np.ndarray | None = None,
    lift_position: float | None = None,
    shoulder_lift_seed: float | None = 1.0,
    ori_weight: float = 0.06,
    max_iters: int = 200,
    tol_pos: float = 0.006,
    tol_ori: float = 0.10,
    damping: float = 0.05,
    fd_eps: float = 1e-3,
    max_joint_step: float = 0.10,
) -> IKResult:
    """Full-pose IK: solve ALL arm joints for TCP position + gripper approach axis +
    jaw-separation axis. Needed for the 6-DOF Pro arm, whose last joint is not a pure
    tool roll, so position-then-wrist_roll (the 5-DOF trick) throws the TCP off target.
    Residual = [pos(3); w*(approach_des - approach_cur)(3); w*(jaw_des - jaw_cur)(3)].
    """
    target = np.asarray(target, dtype=np.float64).reshape(3)
    use_appr = approach_dir is not None
    a_des = None
    if use_appr:
        a_des = np.asarray(approach_dir, np.float64).reshape(3)
        a_des = a_des / (np.linalg.norm(a_des) + 1e-12)
    j_des = np.asarray(jaw_dir, np.float64).reshape(3)
    j_des = j_des / (np.linalg.norm(j_des) + 1e-12)

    env_unwrapped = getattr(env, "unwrapped", env)
    agent = env_unwrapped.agent
    robot = agent.robot
    qpos0 = get_active_qpos(robot)
    names = [joint.name for joint in robot.active_joints]
    index = {name: i for i, name in enumerate(names)}
    arm_indices = [index[name] for name in _arm_joint_names(agent, arm)]
    n = len(arm_indices)
    lo, hi = _arm_joint_limits(robot, arm_indices)

    q_work = qpos0.copy()
    if lift_position is not None:
        for name in getattr(agent, "lift_joint_names", []):
            if name in index:
                q_work[index[name]] = float(lift_position)
    if seed is None:
        q_arm = q_work[arm_indices].astype(np.float64)
        if shoulder_lift_seed is not None and q_arm.size > 1:
            q_arm[1] = float(shoulder_lift_seed)
    else:
        q_arm = np.asarray(seed, dtype=np.float64).reshape(n)
    q_arm = np.clip(q_arm, lo, hi)

    def residual():
        pos = tcp_position(agent, arm)
        j_cur = jaw_axis(agent, arm)
        # jaw axis is a line (sign-free): flip to match desired hemisphere
        if float(np.dot(j_cur, j_des)) < 0:
            j_cur = -j_cur
        e_pos = target - pos
        parts = [e_pos, ori_weight * (j_des - j_cur)]
        ori_err = float(np.linalg.norm(j_des - j_cur))
        if use_appr:
            a_cur = gripper_approach_axis(agent, arm)
            parts.insert(1, ori_weight * (a_des - a_cur))
            ori_err += float(np.linalg.norm(a_des - a_cur))
        return np.concatenate(parts), float(np.linalg.norm(e_pos)), ori_err

    best_qpos = q_work.copy(); best_score = float("inf"); best_pos = float("inf"); best_ori = float("inf")
    success = False; iterations = 0
    with temporary_robot_qpos(robot):
        for iterations in range(1, max_iters + 1):
            q_work[arm_indices] = q_arm
            set_active_qpos(robot, q_work)
            r, pos_err, ori_err = residual()
            score = pos_err + 0.2 * ori_err
            if score < best_score:
                best_score, best_pos, best_ori = score, pos_err, ori_err
                best_qpos = q_work.copy()
            if pos_err <= tol_pos and ori_err <= tol_ori:
                success = True
                break
            m = r.shape[0]
            jac = np.zeros((m, n), dtype=np.float64)
            for c in range(n):
                q_probe = q_arm.copy(); q_probe[c] += fd_eps
                q_work[arm_indices] = q_probe
                set_active_qpos(robot, q_work)
                r_p, _, _ = residual()
                jac[:, c] = (r_p - r) / fd_eps * -1.0
            lhs = jac @ jac.T + (damping * damping) * np.eye(m)
            try:
                dq = jac.T @ np.linalg.solve(lhs, r)
            except np.linalg.LinAlgError:
                dq = jac.T @ np.linalg.pinv(lhs) @ r
            dq = np.clip(dq, -max_joint_step, max_joint_step)
            q_arm = np.clip(q_arm + dq, lo, hi)

    return IKResult(
        arm=arm, target=target.astype(np.float32),
        qpos=best_qpos.astype(np.float32),
        arm_qpos=best_qpos[arm_indices].astype(np.float32),
        success=success, error=best_pos, iterations=iterations,
        wrist_roll=None, wrist_roll_score=best_ori,
    )


def apply_horizontal_jaw_wrist_roll(
    env: Any,
    result: IKResult,
    *,
    roll_min: float = -3.0,
    roll_max: float = 3.0,
    samples: int = 121,
) -> IKResult:
    """Set wrist_roll so the two finger tips separate horizontally."""

    env_unwrapped = getattr(env, "unwrapped", env)
    agent = env_unwrapped.agent
    robot = agent.robot
    names = [joint.name for joint in robot.active_joints]
    index = {name: i for i, name in enumerate(names)}
    arm_indices = [index[name] for name in _arm_joint_names(agent, result.arm)]
    wrist_index = arm_indices[-1]  # last arm joint acts as the tool roll (5-DOF or 6-DOF)

    best_roll = float(result.qpos[wrist_index])
    best_score = float("inf")
    q_probe = np.asarray(result.qpos, dtype=np.float32).copy()

    with temporary_robot_qpos(robot):
        for roll in np.linspace(roll_min, roll_max, max(2, int(samples))):
            q_probe[wrist_index] = float(roll)
            set_active_qpos(robot, q_probe)
            z1, z2 = finger_tip_z(agent, result.arm)
            score = abs(z2 - z1)
            if score < best_score:
                best_score = score
                best_roll = float(roll)

    q_out = np.asarray(result.qpos, dtype=np.float32).copy()
    q_out[wrist_index] = best_roll
    arm_qpos = q_out[arm_indices].astype(np.float32)
    return IKResult(
        arm=result.arm,
        target=result.target,
        qpos=q_out,
        arm_qpos=arm_qpos,
        success=result.success,
        error=result.error,
        iterations=result.iterations,
        wrist_roll=best_roll,
        wrist_roll_score=best_score,
    )


def resolve_actor(env: Any, name: str) -> Any:
    env_unwrapped = getattr(env, "unwrapped", env)
    if hasattr(env_unwrapped, name):
        return getattr(env_unwrapped, name)
    scene = getattr(env_unwrapped, "scene", None)
    if scene is not None:
        actors = getattr(scene, "actors", None)
        if isinstance(actors, dict) and name in actors:
            return actors[name]
        get_actor = getattr(scene, "get_actor", None)
        if callable(get_actor):
            try:
                return get_actor(name)
            except Exception:
                pass
    raise KeyError(f"Could not resolve actor {name!r} from the environment.")


def actor_position(actor: Any) -> np.ndarray:
    pose = getattr(actor, "pose", None)
    if pose is None or not hasattr(pose, "p"):
        raise ValueError(f"Actor {actor!r} does not expose pose.p.")
    pos = to_numpy(pose.p)
    if pos.ndim == 2:
        pos = pos[0]
    return np.asarray(pos, dtype=np.float32).reshape(-1)[:3]


def get_active_qpos(robot: Any) -> np.ndarray:
    qpos = to_numpy(robot.get_qpos())
    if qpos.ndim == 2:
        qpos = qpos[0]
    return np.asarray(qpos, dtype=np.float32).reshape(-1)


def set_active_qpos(robot: Any, qpos: np.ndarray) -> None:
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    current = robot.get_qpos()
    current_np = to_numpy(current)
    if current_np.ndim == 2:
        qpos_out = np.repeat(qpos[None, :], current_np.shape[0], axis=0)
    else:
        qpos_out = qpos
    if hasattr(current, "detach"):
        import torch

        tensor = torch.as_tensor(qpos_out, dtype=current.dtype, device=current.device)
        robot.set_qpos(tensor)
    else:
        robot.set_qpos(qpos_out)


class temporary_robot_qpos:
    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self.qpos = get_active_qpos(robot)

    def __enter__(self) -> "temporary_robot_qpos":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        set_active_qpos(self.robot, self.qpos)


def tcp_position(agent: Any, arm: str) -> np.ndarray:
    attr = "tcp_pos" if arm == "left" else "tcp_pos_2"
    pos = to_numpy(getattr(agent, attr))
    if pos.ndim == 2:
        pos = pos[0]
    return np.asarray(pos, dtype=np.float64).reshape(-1)[:3]


def finger_tip_z(agent: Any, arm: str) -> tuple[float, float]:
    prefix = "left" if arm == "left" else "right"
    tip1 = getattr(agent, f"{prefix}_finger1_tip")
    tip2 = getattr(agent, f"{prefix}_finger2_tip")
    p1 = to_numpy(tip1.pose.p)
    p2 = to_numpy(tip2.pose.p)
    if p1.ndim == 2:
        p1 = p1[0]
    if p2.ndim == 2:
        p2 = p2[0]
    return float(p1.reshape(-1)[2]), float(p2.reshape(-1)[2])


def _arm_joint_limits(robot: Any, arm_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) joint limits for the given active-joint indices. Infinite/absent
    limits fall back to +-2*pi so clipping is a no-op for unlimited joints."""
    try:
        qlim = to_numpy(robot.get_qlimits())
    except Exception:
        qlim = None
    if qlim is None:
        big = np.full(len(arm_indices), 2 * np.pi)
        return -big, big
    if qlim.ndim == 3:
        qlim = qlim[0]
    lo = qlim[arm_indices, 0].astype(np.float64)
    hi = qlim[arm_indices, 1].astype(np.float64)
    lo = np.where(np.isfinite(lo), lo, -2 * np.pi)
    hi = np.where(np.isfinite(hi), hi, 2 * np.pi)
    return lo, hi


def _arm_joint_names(agent: Any, arm: str) -> list[str]:
    if arm == "left":
        return list(getattr(agent, "left_arm_joint_names"))
    if arm == "right":
        return list(getattr(agent, "right_arm_joint_names"))
    raise ValueError(f"Unsupported arm {arm!r}; expected 'left' or 'right'.")


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)

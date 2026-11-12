"""Feedback refresh and protection recovery for AlohaMini evaluation."""

import time
from copy import deepcopy


def safety_snapshot(robot):
    status = deepcopy(getattr(robot, "latest_safety_status", {}))
    if status:
        received_at = getattr(robot, "_last_safety_received_at", None)
        status["feedback_age_s"] = None if received_at is None else time.monotonic() - received_at
    return status


def hold_action(observation):
    action = {key: value for key, value in observation.items() if key.endswith(".pos")}
    action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    if "lift_axis.height_mm" in observation:
        action["lift_axis.height_mm"] = observation["lift_axis.height_mm"]
    return action


def stop_inference(engine):
    """Require an idle worker before resetting the policy or starting a new episode."""
    engine.stop()
    thread = getattr(engine, "_rtc_thread", None)
    if thread is not None and thread.is_alive():
        raise RuntimeError("RTC inference is still running; refusing to reset or restart it")


class EvaluationSafetyGuard:
    """Separate control availability, active protection, and prediction invalidation."""

    def __init__(self):
        self._host_id = None
        self._joint_hold_events = 0

    @property
    def context(self):
        return self._host_id, self._joint_hold_events

    def acknowledge(self, status):
        self._host_id = status["host_session_id"]
        self._joint_hold_events = status["joint_hold_events"]

    def reason(self, robot):
        status = getattr(robot, "latest_safety_status", {})
        received_at = getattr(robot, "_last_safety_received_at", None)
        if status.get("version") != 1:
            return "Host 未提供保护状态，请更新树莓派 Host"
        if status.get("software_stop", {}).get("latched"):
            return "Host 已锁定停止，请检查现场并手动重新启动 Host"
        if not getattr(robot, "command_permitted", True):
            return "控制权由其他客户端持有"
        if not getattr(robot, "control_feedback_valid", True):
            return "Host 反馈中断或过期"
        feedback_timeout = status.get("command_watchdog_timeout_s", 1.0)
        if received_at is None or time.monotonic() - received_at > feedback_timeout:
            return "Host 反馈中断"
        if self._host_id is not None and self._host_id != status["host_session_id"]:
            return "Host 已重新启动"
        self.acknowledge(status)
        if status["joint_holds"]:
            return "关节保护：" + ", ".join(status["joint_holds"])
        return None

    def check_observation(self, robot, observation, *, refresh=False):
        """Refresh after blocking inference or prolonged loss, not on a dataset age limit."""
        if refresh or not getattr(robot, "control_feedback_valid", True):
            observation = robot.refresh_observation()
        reason = self.reason(robot)
        if reason not in (None, "Host 反馈中断或过期", "Host 反馈中断") and not reason.startswith(
            "关节保护："
        ):
            raise RuntimeError(f"{reason}；评估停止。")
        return observation, reason

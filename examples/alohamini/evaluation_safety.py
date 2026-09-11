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
    """Pause autonomous motion on protection or loss of fresh Host feedback."""

    def __init__(self):
        self._host_id = None
        self._events = (0, 0)

    def acknowledge(self, status):
        self._host_id = status["host_session_id"]
        self._events = (status["joint_hold_events"], status["watchdog_events"])

    def reason(self, robot):
        status = getattr(robot, "latest_safety_status", {})
        received_at = getattr(robot, "_last_safety_received_at", None)
        if status.get("version") != 1:
            return "Host 未提供保护状态，请更新树莓派 Host"
        if not getattr(robot, "command_permitted", True):
            return "控制权由其他客户端持有"
        if not getattr(robot, "feedback_fresh", True):
            return "Host 反馈中断或过期"
        feedback_timeout = min(1.0, status.get("command_watchdog_timeout_s", 1.0))
        if received_at is None or time.monotonic() - received_at > feedback_timeout:
            return "Host 反馈中断"
        if self._host_id is not None and self._host_id != status["host_session_id"]:
            return "Host 已重新启动"
        if status["joint_holds"]:
            return "关节保护：" + ", ".join(status["joint_holds"])
        if status["watchdog_active"]:
            return "Host 命令超时保护"
        events = (status["joint_hold_events"], status["watchdog_events"])
        if self._host_id is not None and events != self._events:
            return "上个反馈周期内发生过保护"
        self.acknowledge(status)
        return None

    def check_observation(self, robot, observation):
        """Refresh an expired snapshot before deciding whether feedback is lost."""
        if not robot.feedback_fresh:
            observation = robot.refresh_observation()
        return observation, self.reason(robot)

    def recover(self, robot, engine, interpolator, recorder, observation, reason):
        engine.pause()
        interpolator.reset()
        # The PC command does not wait for a Host acknowledgement.
        robot.send_action(hold_action(observation))
        if recorder is not None:
            recorder.write(
                safety=safety_snapshot(robot), event={"type": "evaluation_paused", "reason": reason}
            )
        stop_inference(engine)
        while True:
            input(f"{reason}。请排除障碍并用遥操反向解除关节保持；按 Enter 重新检查并恢复，Ctrl+C 结束：")
            # Discard pre-pause responses before accepting a recovery snapshot.
            previous = robot.observation_sequence
            observation = robot.refresh_observation()
            deadline = time.monotonic() + 2.0
            while robot.observation_sequence == previous and time.monotonic() < deadline:
                observation = robot.get_observation()
            status = safety_snapshot(robot)
            if robot.observation_sequence == previous or status.get("version") != 1:
                reason = "尚未收到新的保护状态"
                continue
            if not robot.feedback_fresh or not robot.command_permitted:
                reason = "反馈仍不可用或控制权由其他客户端持有"
                continue
            if status["joint_holds"]:
                reason = "关节保持尚未解除"
                continue
            # Explicit recovery acknowledges an idle watchdog with a fresh hold command.
            robot.send_action(hold_action(observation))
            command = dict(robot.last_sent_command)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                observation = robot.get_observation()
                status = safety_snapshot(robot)
                if status.get("command") == command and not status.get("watchdog_active"):
                    break
            else:
                reason = "Host 未确认恢复命令"
                continue
            if status["joint_holds"]:
                reason = "关节保持尚未解除"
                continue
            self.acknowledge(status)
            engine.reset()
            engine.start()
            engine.resume()
            if recorder is not None:
                recorder.write(safety=status, event={"type": "evaluation_resumed"})
            return

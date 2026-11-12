"""通过观察通道锁定停止，避免控制通道的单写者与覆盖队列丢弃停止请求。"""

import math
import re


class SoftwareStop:
    def __init__(self):
        self.latched = False
        self.applied = False
        self.request_id = None
        self.error = None

    def request(self, token, host_session_id):
        if not isinstance(token, bytes):
            return
        parts = token.split(b":")
        if len(parts) != 4 or parts[0] != b"studio-stop" or parts[3] != b"state":
            return
        if parts[1] != host_session_id.encode() or re.fullmatch(b"[a-f0-9]{32}", parts[2]) is None:
            return
        self.latched = True
        self.request_id = parts[2].decode()

    def apply(self, robot, observation, owner):
        if not self.latched or self.applied:
            return {}
        try:
            try:
                action = {key: float(value) for key, value in observation.items() if key.endswith(".pos")}
                action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
                if not all(math.isfinite(value) for value in action.values()):
                    raise ValueError("停止反馈包含非有限位置")
                robot.send_action(action)
            finally:
                robot.stop_motion()
        except Exception as error:
            self.error = str(error)
            return {}
        owner.release()
        self.applied = True
        self.error = None
        return action

    def status(self):
        return {"latched": self.latched, "applied": self.applied, "request_id": self.request_id, "error": self.error}

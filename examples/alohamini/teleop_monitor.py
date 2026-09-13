"""Low-rate operator feedback without additional robot reads or commands."""

import math
import time


def tracking_rows(robot):
    """Compare Host targets with its sampled positions in their original command units."""
    safety = robot.latest_safety_status
    targets = safety.get("requested_targets", {})
    currents = safety.get("currents_ma", {})
    motors = robot.latest_robot_metadata.get("motors", {})
    rows = []
    for key, command in sorted(safety.get("accepted_targets", {}).items()):
        if not key.endswith(".pos") or key not in robot.last_remote_state:
            continue
        measured = float(robot.last_remote_state[key])
        target = float(targets.get(key, command))
        command = float(command)
        if not all(math.isfinite(value) for value in (measured, target, command)):
            continue
        motor = key.removesuffix(".pos")
        current = currents.get(motor)
        current_text = "n/a" if current is None else f"{float(current):+.1f}mA"
        unit = motors.get(motor, {}).get("normalization", "command_units")
        rows.append(
            (
                motor,
                unit,
                abs(command - measured),
                (
                    f"[TELEOP TRACKING][{motor}] target={target:.2f} command={command:.2f} "
                    f"measured={measured:.2f} gap={command - measured:+.2f} unit={unit} current={current_text}"
                ),
            )
        )
    return rows


class TeleopMonitor:
    """Print rates and largest gaps per arm/unit at most once per second."""

    def __init__(self, robot):
        self.robot = robot
        self._started_at = time.perf_counter()
        self._cycles = 0
        self._sent = 0

    def update(self, *, sent: bool):
        self._cycles += 1
        self._sent += int(sent)
        now = time.perf_counter()
        elapsed = now - self._started_at
        if elapsed < 1.0:
            return
        fresh = self.robot.feedback_fresh
        permitted = self.robot.command_permitted
        safety = self.robot.latest_safety_status
        state = (
            "active" if fresh and permitted else "waiting_feedback" if not fresh else "control_unavailable"
        )
        if fresh and permitted and not self._sent:
            state = "waiting_send"
        holds = len(safety.get("joint_holds", {})) if fresh else "n/a"
        print(
            f"[TELEOP] loop_hz={self._cycles / elapsed:.1f} sent_hz={self._sent / elapsed:.1f} "
            f"state={state} joint_holds={holds}",
            flush=True,
        )
        if fresh:
            rows = tracking_rows(self.robot)
            for prefix in ("arm_left_", "arm_right_"):
                arm_rows = [row for row in rows if row[0].startswith(prefix)]
                # Grippers and arm joints may use different normalization ranges.
                for unit in sorted({row[1] for row in arm_rows}):
                    group = [row for row in arm_rows if row[1] == unit]
                    print(max(group, key=lambda row: row[2])[3], flush=True)
        self._started_at = now
        self._cycles = 0
        self._sent = 0

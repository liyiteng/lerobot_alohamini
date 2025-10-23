import sys
import os
import argparse
import time
import ast
import json

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

DEFAULT_PORT = "/dev/ttyACM0"
HALF_TURN_DEGREE = 180

GENERAL_ACTIONS = {"sleep", "print"}
DEFAULT_FEETECH_MODEL = "sts3215"
SCAN_START = 1
SCAN_END = 22


# --------------------------- Probe / Scan helpers (reusable) --------------------------- #

def probe_scan_ids(port: str) -> dict[int, str]:
    """
    探针扫描 ID 范围，返回 {id: model_name}（只含在线电机）。
    仅做 ping，不对电机写寄存器；断开时不关力矩。
    """

    probe_bus = build_bus(port, {})
    try:
        probe_bus.connect(handshake=False)
    except Exception as e:
        print(f"Scan connect failed: {e}")
        return {}

    found: dict[int, str] = {}
    try:
        for mid in range(int(SCAN_START), int(SCAN_END) + 1):
            try:
                model_nb = probe_bus.ping(mid, num_retry=1)
                if model_nb is not None:
                    try:
                        model_name = probe_bus._model_nb_to_model(model_nb)
                    except Exception:
                        model_name = str(model_nb)
                    found[mid] = model_name
            except Exception:
                continue
    finally:
        try:
            probe_bus.disconnect(disable_torque=False)
        except Exception:
            pass
    return found


def build_motors_from_scan(port: str):
    """
    基于 probe_scan_ids 结果，构建 motors 字典（名称统一为 motor_<id>）。
    """
    from lerobot.motors.motors_bus import Motor, MotorNormMode
    found = probe_scan_ids(port)
    if not found:
        return {}
    motors = {
        f"motor_{mid}": Motor(id=mid, model=model, norm_mode=MotorNormMode.RANGE_0_100)
        for mid, model in found.items()
    }
    return motors





def parse_actions(file_path):
    actions = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            actions.append(parts)
    return actions


def evaluate_expression(expression, variables):
    """
    expression evaluation, supporting {variables} and +/− operations.
    e.g. "{Present_Position} + 500"
    """
    for var, value in variables.items():
        expression = expression.replace(f"{{{var}}}", str(value))
    try:
        return eval(expression)
    except Exception as e:
        print(f"Error evaluating expression '{expression}': {e}")
        return None


def build_bus(port, motors):
    return FeetechMotorsBus(port=port, motors=motors)


# --------------------------- High‑level helpers --------------------------- #

def _connect_bus(bus):
    try:
        bus.connect(handshake=False)
        print(f"Connected on port {bus.port}")
        return True
    except OSError as e:
        print(f"Error occurred when connecting to the motor bus: {e}")
        return False


def _motor_angle_from_position(position):
    return position / (4096 // 2) * HALF_TURN_DEGREE

# -------------------------------------------------------------- #
# ---------------------------Function--------------------------- #
# -------------------------------------------------------------- #

def get_motors_states(port):
    """
    动态表格显示电机状态；总是先扫描 [scan_start, scan_end]，只显示在线电机。
    - 传了 motors 也不会直接用，而是按 ID 范围扫描后重建 motors（保证不读离线 ID）
    """
    import time, sys, shutil
    from lerobot.motors.motors_bus import Motor, MotorNormMode

    # ---------- ANSI 工具 ----------
    CSI = "\x1b["
    def _hide_cursor(): sys.stdout.write(f"{CSI}?25l"); sys.stdout.flush()
    def _show_cursor(): sys.stdout.write(f"{CSI}?25h"); sys.stdout.flush()
    def _move_up(n):    sys.stdout.write(f"{CSI}{n}A") if n>0 else None; sys.stdout.flush()
    def _clear_line():  sys.stdout.write(f"{CSI}2K\r")
    def _term_width():
        try: return max(60, min(200, shutil.get_terminal_size().columns))
        except Exception: return 100
    def _format_row(name: str, st: dict, maxw: int) -> str:
        def F(v, w, a=">"):
            s = "-" if v is None else str(v)
            if len(s) > w: s = s[:w]
            return f"{s:{a}{w}}"
        row = (f"{F(name,15,'<')} | {F(st.get('ID'),3)} | {F(st.get('Position'),6)} | "
            f"{F(st.get('Offset'),6)} | {F(st.get('Angle'),6)} | "
            f"{F(st.get('Load'),6)} | {F(st.get('Acceleration'),6)} | "
            f"{F(st.get('Voltage'),4)} | {F(st.get('Current(mA)'),8)} | "
            f"{F(st.get('Temperature'),4)} | {F(st.get('Port','-'),16,'<')}")
        return row[:maxw] if len(row) > maxw else row

    # ---------- 第一步：用探针扫描在线电机 ----------

    motors = build_motors_from_scan(port)
    if not motors:
        print(f"No motors found in ID range [{SCAN_START}, {SCAN_END}] on {port}.")
        return


    # ---------- 第二步：用“只包含在线电机”的字典进入动态显示 ----------
    bus = build_bus(port, motors)
    if not _connect_bus(bus):  # 你的辅助函数里保持 handshake=False 更稳
        return

    try:
        ids_str = ", ".join(str(m.id) for m in bus.motors.values())
        print(f"Online IDs: [{ids_str}] on {port}")

        printed_lines = 0
        interval_s = 0.1
        _hide_cursor()

        while True:
            try:
                pos  = bus.sync_read("Present_Position",        normalize=False)
                load = bus.sync_read("Present_Load",            normalize=False)
                acc  = bus.sync_read("Maximum_Acceleration",    normalize=False)
                volt = bus.sync_read("Present_Voltage",         normalize=False)
                curr = bus.sync_read("Present_Current",         normalize=False)
                offset = bus.sync_read("Homing_Offset",        normalize=False)
                temp = bus.sync_read("Present_Temperature",     normalize=False)
            except Exception as e:
                print(f"Sync read error: {e}")
                break

            rows = []
            for name in motors.keys():
                try:
                    st = {
                        "ID":           bus.read("ID", name, normalize=False),
                        "Position":     pos.get(name),
                        "Load":         load.get(name),
                        "Acceleration": acc.get(name),
                        "Voltage":      volt.get(name),
                        "Current(mA)":  (curr.get(name) * 6.5) if (curr.get(name) is not None) else None,
                        "Offset":      offset.get(name),
                        "Temperature":  temp.get(name),
                        "Port":         port,
                    }
                    angle = _motor_angle_from_position(st["Position"]) if st["Position"] is not None else None
                    # pos_raw = st["Position"]
                    # off_raw = st["Offset"]
                    # angle = _motor_angle_from_position(pos_raw - off_raw) if pos_raw is not None else None

                    st["Angle"] = None if angle is None else round(angle, 1)
                    rows.append((name, st))
                except Exception:
                    continue



            maxw = _term_width()
            sep  = "-" * min(maxw, 140)  # 由 120 放宽到 140
            header = (f"{'NAME':<15} | {'ID':>3} | {'POS':>6} | {'OFF':>6} | {'ANG':>6} | "
                    f"{'LOAD':>6} | {'ACC':>6} | {'VOLT':>4} | {'CURR(MA)':>8} | {'TEMP':>4} | PORT")
            header = header[:maxw] if len(header) > maxw else header
            frame_lines = [sep, header] + [_format_row(n, s, maxw) for (n, s) in rows] + \
                          [sep, f"Updated: {time.strftime('%H:%M:%S')}   (Ctrl+C 退出)"]

            _move_up(printed_lines)
            for ln in frame_lines:
                _clear_line(); sys.stdout.write(ln + "\n")

            extra = printed_lines - len(frame_lines)
            for _ in range(max(0, extra)):
                _clear_line(); sys.stdout.write("\n")

            sys.stdout.flush()
            printed_lines = len(frame_lines)
            time.sleep(interval_s)

    except KeyboardInterrupt:
        pass
    finally:
        # 关键：不要在断开时去“对所有电机”关力矩，避免离线 ID 报错
        try:
            bus.disconnect(disable_torque=False)
        finally:
            _show_cursor()



def configure_motor_id(port: str, current_id: int, new_id: int):

    current_id = int(current_id)
    new_id = int(new_id)
    if current_id == new_id:
        raise SystemExit("current_id == new_id，没有必要改。")

    # 1) 先用“裸总线”去 ping 校验（不带 motors 映射，避免库做多余事）
    probe_bus = FeetechMotorsBus(port=port, motors={})
    try:
        probe_bus.connect(handshake=False)
        ok_cur = probe_bus.ping(current_id, num_retry=1) is not None
        ok_new = probe_bus.ping(new_id, num_retry=1) is not None
    except Exception as e:
        raise SystemExit(f"Probe connect failed: {e}")
    finally:
        try:
            probe_bus.disconnect(disable_torque=False)
        except Exception:
            pass

    if not ok_cur:
        raise SystemExit(f"[ABORT] 未发现 ID={current_id} 在线，放弃改 ID。")
    if ok_new:
        raise SystemExit(f"[ABORT] 目标 ID={new_id} 已被占用，放弃改 ID。")

    # 2) 用“只包含 current_id 的单电机字典”建立总线（名字无所谓，只要 id 是 current_id）
    tmp_name = f"motor_{current_id}"
    one_motor = {
        tmp_name: Motor(id=current_id, model=DEFAULT_FEETECH_MODEL, norm_mode=MotorNormMode.RANGE_0_100)
    }

    bus = FeetechMotorsBus(port=port, motors=one_motor)
    try:
        bus.connect(handshake=False)
        print(f"Connected on port {bus.port} (current_id={current_id})")

        # 保险：解锁 & 力矩处理（按你电机需要，可调整/删掉）
        try:
            bus.write("Lock", tmp_name, 0, normalize=False)
        except Exception:
            pass
        try:
            bus.disable_torque(tmp_name)
        except Exception:
            pass

        # 3) 直接对“当前 id 的那颗”写寄存器：ID = new_id
        bus.write("ID", tmp_name, new_id, normalize=False)
        time.sleep(0.2)

        # 4) 改后校验：当前 id 应该失联，new_id 应该在线
        # 这里用一个“空 bus”去 ping，避免受 motors 映射影响
        probe_bus2 = FeetechMotorsBus(port=port, motors={})
        try:
            probe_bus2.connect(handshake=False)
            ok_cur2 = probe_bus2.ping(current_id, num_retry=1) is not None
            ok_new2 = probe_bus2.ping(new_id, num_retry=1) is not None
        finally:
            try:
                probe_bus2.disconnect(disable_torque=False)
            except Exception:
                pass

        if ok_cur2 or not ok_new2:
            raise SystemExit(
                f"[VERIFY FAIL] 改后验证失败：ID={current_id} 仍在线={ok_cur2}, ID={new_id} 在线={ok_new2}"
            )

        print(f"[OK] Changed ID {current_id} -> {new_id} (model={DEFAULT_FEETECH_MODEL})")

    except Exception as e:
        print(f"Error during motor ID configuration: {e}")
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


def reset_motors_to_midpoint(port):

    motors = build_motors_from_scan(port)
    if not motors:
        print(f"No motors found in ID range [{SCAN_START}, {SCAN_END}] on {port}.")
        return


    bus = build_bus(port, motors)
    if not _connect_bus(bus):
        return

    try:
        for name in motors:
            try:
                bus.write("Torque_Enable", name, 128, normalize=False)
                time.sleep(0.1)
                bus.write("Lock", name, 0, normalize=False)
                bus.write("Maximum_Acceleration", name, 254, normalize=False)
                bus.write("Acceleration", name, 254, normalize=False)
                print(f"------- {name} to midpoint config complete!------")
            except Exception as motor_e:
                print(f"Error on '{name}': {motor_e}")
                continue
    except Exception as e:
        print(f"Error during midpoint config: {e}")
    finally:
        bus.disconnect()


def reset_motors_torque(port):

    motors = build_motors_from_scan(port)
    if not motors:
        print(f"No motors found in ID range [{SCAN_START}, {SCAN_END}] on {port}.")
        return

    bus = build_bus(port, motors)
    if not _connect_bus(bus):
        return
    try:
        for name in motors:
            try:
                _ = bus.read("ID", name, normalize=False)
                bus.write("Torque_Enable", name, 0, normalize=False)
                print(f"------- {name} reset torque complete!------")
            except Exception as motor_e:
                print(f"Error on '{name}': {motor_e}")
                continue
    except Exception as e:
        print(f"Error during torque reset: {e}")
    finally:
        bus.disconnect()




def move_motor_to_position(
    port: str,
    motor_id: int,
    position: int,
    ):
    """
    直接用数值ID控制电机到指定 position（原始ticks）。
    - 不需要 motors{}，不需要电机名。
    - 默认确保处于“位置模式”，必要时切换。
    """
    tmp_name = f"motor_{int(motor_id)}"
    one_motor = {
        tmp_name: Motor(id=int(motor_id), model=DEFAULT_FEETECH_MODEL, norm_mode=MotorNormMode.RANGE_0_100)
    }

    # 可选：简单范围夹取（12bit 0~4095；按你家实际寄存器范围调整）
    try:
        position = int(position)
        position = max(0, min(4095, position))
    except Exception:
        raise SystemExit(f"Invalid --position: {position}")

    bus = FeetechMotorsBus(port=port, motors=one_motor)
    try:
        bus.connect(handshake=False)
        print(f"Connected on port {bus.port} (ID={motor_id})")

        # 若不确定电机当前模式，保险起见切到“位置模式”
        bus.disable_torque(tmp_name)
        bus.write("Operating_Mode", tmp_name, OperatingMode.POSITION.value, normalize=False)
        bus.enable_torque(tmp_name)

        pre = bus.read("Present_Position", tmp_name, normalize=False)
        bus.write("Goal_Position", tmp_name, position, normalize=False)

        print(f"[ID {motor_id}] Present_Position(before)={pre}")
        print(f"[ID {motor_id}] → Goal_Position={position}")
        time.sleep(2.0)

    except Exception as e:
        print(f"Error moving ID {motor_id}: {e}")
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


def move_motors_by_code(port):

    motors = {
        "gripper": Motor(8, DEFAULT_FEETECH_MODEL, MotorNormMode.RANGE_0_100),
        # "wrist_roll": Motor(6, args.model, MotorNormMode.DEGREES),
        # "wrist_yaw": Motor(5, args.model, MotorNormMode.DEGREES),
        # "wrist_flex": Motor(4, args.model, MotorNormMode.DEGREES),
        # "elbow_flex": Motor(3, args.model, MotorNormMode.DEGREES),
        # "shoulder_lift": Motor(2, args.model, MotorNormMode.DEGREES),
        # "shoulder_pan": Motor(1, args.model, MotorNormMode.DEGREES),
    }


    bus = build_bus(port, motors)
    if not _connect_bus(bus):
        return

    try:
        # demo: open/close gripper around current pos
        if "gripper" in motors:
            pos = bus.read("Present_Position", "gripper", normalize=False)
            print(f"gripper Present_Position: {pos}")

            bus.write("Goal_Position", "gripper", pos + 500, normalize=False)
            time.sleep(1)
            p = bus.read("Present_Position", "gripper", normalize=False)
            print(f"+500 position: {p}")

            bus.write("Goal_Position", "gripper", p - 1000, normalize=False)
            time.sleep(1)
            p = bus.read("Present_Position", "gripper", normalize=False)
            print(f"-1000 position: {p}")

            bus.write("Goal_Position", "gripper", p + 500, normalize=False)
            time.sleep(1)
            p = bus.read("Present_Position", "gripper", normalize=False)
            print(f"+500 position: {p}")

            print("start_reset")
            bus.write("Goal_Position", "gripper", 2048, normalize=False)
            time.sleep(2)
            p = bus.read("Present_Position", "gripper", normalize=False)
            print(f"reset_position: {p}")
    except Exception as e:
        print(f"Error during motor demo: {e}")
    finally:
        bus.disconnect()



def move_motors_by_script(port,script_path ):

    motors = {
        "gripper": Motor(8, DEFAULT_FEETECH_MODEL, MotorNormMode.RANGE_0_100),
        # "wrist_roll": Motor(6, args.model, MotorNormMode.DEGREES),
        # "wrist_yaw": Motor(5, args.model, MotorNormMode.DEGREES),
        # "wrist_flex": Motor(4, args.model, MotorNormMode.DEGREES),
        # "elbow_flex": Motor(3, args.model, MotorNormMode.DEGREES),
        # "shoulder_lift": Motor(2, args.model, MotorNormMode.DEGREES),
        # "shoulder_pan": Motor(1, args.model, MotorNormMode.DEGREES),
    }
        

    # Resolve script path relative to this file for convenience
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, script_path)
    print(f"script_path:{script_path}")
    print(f"script_dir:{script_dir}")
    print(f"file_path:{file_path}")

    bus = build_bus(port, motors)
    if not _connect_bus(bus):
        return

    variables = {}
    actions = parse_actions(file_path)
    print(f"actions: {actions}")

    try:
        for action in actions:
            if not action:
                continue

            first = action[0]

            # General actions (sleep/print)
            if first in GENERAL_ACTIONS:
                if first == "sleep":
                    if len(action) < 2:
                        print(f"Invalid sleep action: {action}")
                        continue
                    try:
                        duration = float(action[1])
                        print(f"Sleeping for {duration} seconds")
                        time.sleep(duration)
                    except ValueError:
                        print(f"Invalid duration for sleep action: {action[1]}")
                elif first == "print":
                    if len(action) < 2:
                        print(f"Invalid print action: {action}")
                        continue
                    try:
                        message = action[1].format(**variables)
                    except Exception:
                        message = action[1]
                    print(message)
                continue

            # Motor actions
            motor_name = action[0]
            if motor_name not in motors:
                print(f"Motor '{motor_name}' not found. Skipping action.")
                continue

            if len(action) < 2:
                print(f"Invalid action (missing type): {action}")
                continue

            action_type = action[1]
            print(f"motor_name:{motor_name}, action_type:{action_type}")

            try:
                if action_type == "read":
                    if len(action) < 3:
                        print(f"Invalid read action: {action}")
                        continue
                    register = action[2]
                    value = bus.read(register, motor_name, normalize=False)
                    print(f"{motor_name} - {register}: {value}")
                    variables[register] = value

                elif action_type == "write":
                    if len(action) < 4:
                        print(f"Invalid write action: {action}")
                        continue
                    register = action[2]
                    value_expr = action[3]
                    value = evaluate_expression(value_expr, variables)
                    if value is not None:
                        bus.write(register, motor_name, value, normalize=False)
                        print(f"Set {motor_name} - {register} to {value}")
                else:
                    print(f"Unknown action type '{action_type}' for motor '{motor_name}'. Skipping.")
            except Exception as motor_e:
                print(f"Error performing '{action_type}' on '{motor_name}': {motor_e}")
                continue
    except Exception as e:
        print(f"Error occurred during executing actions: {e}")
    finally:
        bus.disconnect()
        print(f"Disconnected from port {port}")
    print(f"Completed executing actions from {script_path}")







# --------------------------- CLI --------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor utilities (LeRobot latest API)")

    parser.add_argument(
        "command",
        choices=[
            "get_motors_states",
            "configure_motor_id",
            "reset_motors_to_midpoint",
            "reset_motors_torque",
            "move_motor_to_position",
            "move_motors_by_code",
            "move_motors_by_script",
        ],
        help="Command to execute",
    )

    parser.add_argument("--port", type=str, default=DEFAULT_PORT, help=f"Set the port (default: {DEFAULT_PORT})")
    parser.add_argument("--id", type=str, help="Motor name or CURRENT numeric ID (context depends on subcommand)")
    parser.add_argument("--set_id", type=int, help="Desired numeric ID value to set on the motor")
    parser.add_argument("--position", type=str, help="Goal position (raw ticks) for move_motor_to_position")
    parser.add_argument("--script_path", type=str, help="Relative path to CSV-like action script")
    #parser.add_argument("--file", type=str, help="Path to calibration JSON file")

    args = parser.parse_args()
    cmd = args.command

    # ---- helpers for dispatch ----

    commands = {
        "get_motors_states":             lambda: get_motors_states(args.port),

        "configure_motor_id":            lambda: (
            configure_motor_id(args.port, int(args.id), int(args.set_id))
            if (args.id is not None and args.set_id is not None)
            else (_ for _ in ()).throw(SystemExit("--id (current numeric ID) and --set_id are required"))
        ),

        "reset_motors_to_midpoint":      lambda: reset_motors_to_midpoint(args.port),
        "reset_motors_torque":           lambda: reset_motors_torque(args.port),
        "move_motor_to_position":        lambda: (
            move_motor_to_position(args.port,  args.id, args.position)
            if (args.id is not None and args.position is not None)
            else (_ for _ in ()).throw(SystemExit("--id (motor name or numeric ID) and --position are required"))
        ),
        "move_motors_by_code":           lambda: move_motors_by_code(args.port),
        "move_motors_by_script":         lambda: (
            move_motors_by_script(args.port,args.script_path )
            if args.script_path
            else (_ for _ in ()).throw(SystemExit("--script_path is required for move_motors_by_script"))
        ),


    }

    # ---- execute ----
    fn = commands.get(cmd)
    if not fn:
        raise SystemExit(f"Unknown command: {cmd}")
    fn()
#!/usr/bin/env python
"""Studio 的只读采集检查与有反馈的软件停止入口；不连接本机电机。"""

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path

import zmq


def request_observation(socket, token, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    socket.send(token, flags=zmq.NOBLOCK)
    while time.monotonic() < deadline:
        if not socket.poll(max(1, int((deadline - time.monotonic()) * 1000))):
            break
        parts = socket.recv_multipart()
        if len(parts) >= 2 and parts[0] == token:
            observation = json.loads(parts[1])
            if not isinstance(observation, dict):
                raise ValueError("Host 返回了无效状态")
            return observation, parts[2:]
    raise TimeoutError("机器人未返回有效状态，请检查网络与 Host 服务")


def validate_host(observation, robot_model):
    metadata = observation.get("_robot_metadata", {})
    safety = observation.get("_safety", {})
    if metadata.get("robot_model") != robot_model:
        raise ValueError("机器人型号与当前任务不一致，请检查两端配置")
    if not isinstance(safety.get("host_session_id"), str) or not safety["host_session_id"]:
        raise ValueError("Host 缺少有效会话信息，请更新机器人端软件")
    if safety.get("software_stop", {}).get("latched"):
        raise ValueError("Host 已锁定停止；请检查现场并手动重新启动 Host 后再采集")
    if safety.get("control_owner") is not None:
        raise ValueError("机器人正由其他程序控制，请先结束该程序后重新检查")


def validate_cameras(observation, frames, camera_names):
    import cv2
    import numpy as np

    if len(set(camera_names)) != 2 or len(frames) % 2:
        raise ValueError("需要两路不同的机器人相机，或相机响应格式无效")
    images = {frames[i].decode(): frames[i + 1] for i in range(0, len(frames), 2)}
    timing = observation.get("_host_timing", {})
    captures = timing.get("camera_capture_monotonic_s", {})
    # 采集与观察时间均来自 Host，避免跨设备单调时钟相减。
    host_time = timing.get("host_clock_reference", {}).get("monotonic_s")
    for name in camera_names:
        if name not in observation.get("_robot_metadata", {}).get("cameras", []):
            raise ValueError(f"机器人未启用相机 {name}")
        image = cv2.imdecode(np.frombuffer(images.get(name, b""), dtype=np.uint8), cv2.IMREAD_COLOR) if images.get(name) else None
        capture = captures.get(name)
        if image is None or image.size == 0:
            raise ValueError(f"相机 {name} 未返回可解码画面")
        if not isinstance(capture, (int, float)) or not isinstance(host_time, (int, float)) or not math.isfinite(capture) or not math.isfinite(host_time) or not 0 <= host_time - capture <= 0.5:
            raise ValueError(f"相机 {name} 的画面过期或缺少采集时间")


def validate_calibration(observation, leader_id="so101_leader_bi", calibration_root=None):
    root = calibration_root or Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot")) / "calibration"
    joints = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"}
    for side in ("left", "right"):
        path = root / "teleoperators/so_leader" / f"{leader_id}_{side}.json"
        if not path.is_file():
            alternate_id = "am_leader_bi" if leader_id == "so101_leader_bi" else "so101_leader_bi"
            path = root / "teleoperators/so_leader" / f"{alternate_id}_{side}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{side} 主臂标定文件缺失或损坏，请完成标定") from error
        if not isinstance(data, dict) or not joints.issubset(data):
            raise ValueError(f"{side} 主臂标定缺少关节，请重新标定")
        validate_ranges({name: data[name] for name in joints})
    motors = observation.get("_robot_metadata", {}).get("motors", {})
    positions = [key.removesuffix(".pos") for key in observation if key.endswith(".pos")]
    if len(positions) != 14 or not set(positions).issubset(motors):
        raise ValueError("Host 未提供完整的双从臂标定信息")
    if not all(math.isfinite(float(observation[f"{name}.pos"])) for name in positions):
        raise ValueError("Host 返回了无效关节位置")
    validate_ranges({name: motors[name] for name in positions})


def validate_ranges(motors):
    for name, motor in motors.items():
        low, high = motor.get("range_min"), motor.get("range_max")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (low, high)) or not 0 <= low < high <= 4095:
            raise ValueError(f"关节 {name} 标定行程无效")


def stop_robot(socket, observation):
    safety = observation.get("_safety", {})
    if safety.get("software_stop_supported") is not True:
        return {"success": False, "status": "unconfirmed", "message": "当前 Host 不支持停止确认，请更新机器人端；本地进程已请求结束，机器人状态未确认"}
    session = safety.get("host_session_id")
    if not isinstance(session, str) or not session or ":" in session:
        raise ValueError("Host 会话信息无效，无法确认停止")
    request_id = uuid.uuid4().hex
    token = f"studio-stop:{session}:{request_id}:state".encode()
    stopped, _ = request_observation(socket, token)
    result = stopped.get("_safety", {})
    state = result.get("software_stop", {})
    confirmed = result.get("host_session_id") == session and state.get("request_id") == request_id and state.get("latched") is True and state.get("applied") is True
    return {
        "success": confirmed,
        "status": "confirmed" if confirmed else "unconfirmed",
        "message": "Host 已确认执行停止并锁定控制；电机保持供电。继续使用前请检查现场并手动重新启动 Host" if confirmed else "停止执行结果未确认，请检查机器人并使用现场停止措施",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("network", "cameras", "calibration", "stop"), required=True)
    parser.add_argument("--remote_ip", required=True)
    parser.add_argument("--robot_model", default="alohamini2pro")
    parser.add_argument("--camera", action="append", default=[])
    args = parser.parse_args()
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, 2000)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.RCVHWM, 2)
    try:
        socket.connect(f"tcp://{args.remote_ip}:5556")
        suffix = "camera" if args.action == "cameras" else "state"
        observation, frames = request_observation(socket, f"studio-{uuid.uuid4().hex}:{suffix}".encode())
        if args.action == "stop":
            result = stop_robot(socket, observation)
        else:
            validate_host(observation, args.robot_model)
            if args.action == "cameras":
                validate_cameras(observation, frames, args.camera or ["forward", "wrist_right"])
            elif args.action == "calibration":
                validate_calibration(observation)
            result = {"success": True, "message": "检查通过"}
    except Exception as error:
        result = {"success": False, "status": "unconfirmed", "message": str(error)}
    finally:
        socket.close(linger=0)
        context.term()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

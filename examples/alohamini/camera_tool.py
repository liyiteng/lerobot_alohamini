#!/usr/bin/env python

"""AlohaMini Dual-Camera Diagnostic & Snapshot Tool.

Usage:
    python camera_tool.py --action=list --remote_ip=192.168.10.2
    python camera_tool.py --action=snapshot --remote_ip=192.168.10.2
    python camera_tool.py --action=preview --remote_ip=192.168.10.2
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import zmq
except ImportError:
    zmq = None


def test_zmq_camera_stream(remote_ip: str, timeout_ms: int = 1500) -> dict[str, dict]:
    """Listen on ZMQ port 5556 for frames from forward and wrist_right cameras."""
    if not zmq or not remote_ip:
        return {}

    context = zmq.Context()
    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"camera/")
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)

    addr = f"tcp://{remote_ip.strip()}:5556"
    try:
        sock.connect(addr)
    except Exception as e:
        return {}

    captured = {}
    start_time = time.time()

    # Attempt to capture for up to timeout_ms
    while time.time() - start_time < (timeout_ms / 1000.0) and len(captured) < 2:
        try:
            parts = sock.recv_multipart()
            if len(parts) >= 3:
                topic = parts[0].decode(errors="ignore")
                meta_raw = parts[1].decode(errors="ignore")
                jpg_bytes = parts[2]

                cam_name = topic.replace("camera/", "").strip()
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    meta = {}

                captured[cam_name] = {
                    "source": "zmq",
                    "width": meta.get("width", 640),
                    "height": meta.get("height", 480),
                    "sequence": meta.get("sequence", 0),
                    "bytes": jpg_bytes,
                }
        except zmq.Again:
            break
        except Exception:
            break

    sock.close(linger=0)
    context.term()
    return captured


def scan_local_opencv_cameras(max_idx: int = 4) -> list[int]:
    """Scan local OpenCV camera indices 0..max_idx."""
    if not cv2:
        return []

    available = []
    for idx in range(max_idx):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


def capture_local_camera_frame(idx: int) -> bytes | None:
    """Capture a single JPEG frame from a local camera index."""
    if not cv2:
        return None
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if success:
        return buffer.tobytes()
    return None


def main():
    parser = argparse.ArgumentParser(description="AlohaMini Dual-Camera Diagnostic & Snapshot Tool")
    parser.add_argument(
        "--action",
        type=str,
        default="list",
        choices=["list", "snapshot", "preview"],
        help="Action to execute",
    )
    parser.add_argument("--remote_ip", type=str, default="192.168.10.2", help="Remote Pi Host IP")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    remote_ip = args.remote_ip.strip()

    if args.action == "list":
        zmq_cams = test_zmq_camera_stream(remote_ip, timeout_ms=1000)
        local_cams = scan_local_opencv_cameras(4)

        result = {
            "remote_ip": remote_ip,
            "zmq_5556_connected": len(zmq_cams) > 0,
            "zmq_cameras": list(zmq_cams.keys()),
            "local_cameras": local_cams,
        }

        if args.json:
            print(json.dumps(result))
        else:
            print("=== AlohaMini Camera Discovery ===")
            print(f"Remote IP: {remote_ip}")
            print(f"ZMQ:5556 Connected: {len(zmq_cams) > 0} (Streams: {list(zmq_cams.keys())})")
            print(f"Local OpenCV Cameras: {local_cams}")

    elif args.action == "snapshot":
        # First try Pi ZMQ 5556
        zmq_cams = test_zmq_camera_stream(remote_ip, timeout_ms=1200)
        camera_data = []

        if zmq_cams:
            for name, item in zmq_cams.items():
                b64 = base64.b64encode(item["bytes"]).decode("ascii")
                camera_data.append({
                    "name": name,
                    "source": "zmq",
                    "width": item["width"],
                    "height": item["height"],
                    "base64": b64,
                })
        else:
            # Fallback to local cameras
            local_indices = scan_local_opencv_cameras(4)
            names = ["forward", "wrist_right"]
            for i, idx in enumerate(local_indices[:2]):
                jpg = capture_local_camera_frame(idx)
                if jpg:
                    b64 = base64.b64encode(jpg).decode("ascii")
                    camera_data.append({
                        "name": names[i] if i < len(names) else f"camera_{idx}",
                        "source": f"local_{idx}",
                        "width": 640,
                        "height": 480,
                        "base64": b64,
                    })

        output = {
            "success": len(camera_data) > 0,
            "count": len(camera_data),
            "cameras": camera_data,
        }
        print(json.dumps(output))

    elif args.action == "preview":
        print(f"Testing stream from {remote_ip} for 5 seconds...")
        zmq_cams = test_zmq_camera_stream(remote_ip, timeout_ms=3000)
        if zmq_cams:
            print(f"Successfully received frames from: {list(zmq_cams.keys())}")
        else:
            local_cams = scan_local_opencv_cameras(2)
            print(f"Local cameras available: {local_cams}")


if __name__ == "__main__":
    main()

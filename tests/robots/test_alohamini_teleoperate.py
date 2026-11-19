import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    ("options", "camera_fps"),
    [
        ([], 30),
        (["--camera-fps", "10"], 10),
    ],
)
def test_teleoperation_refreshes_startup_feedback_without_changing_camera_cadence(
    monkeypatch, options, camera_fps
):
    from lerobot.robots import alohamini
    from lerobot.teleoperators import bi_so_leader
    from lerobot.teleoperators.keyboard import teleop_keyboard
    from lerobot.utils import robot_utils, visualization_utils

    robot = Mock(is_connected=True)
    robot.get_observation.return_value = {}
    robot._from_keyboard_to_base_action.return_value = {"x.vel": 0.0}
    robot._from_keyboard_to_lift_action.return_value = {"lift_axis.height_mm": 0.0}
    leader = Mock(is_connected=True)
    leader.get_action.return_value = {}
    keyboard = Mock(is_connected=True)
    keyboard.get_action.return_value = []
    make_client = Mock(return_value=robot)
    monkeypatch.setattr(alohamini, "AlohaMiniClient", make_client)
    monkeypatch.setattr(bi_so_leader, "BiSOLeader", lambda _cfg: leader)
    monkeypatch.setattr(teleop_keyboard, "KeyboardTeleop", lambda _cfg: keyboard)
    monkeypatch.setattr(visualization_utils, "init_rerun", Mock())
    monkeypatch.setattr(visualization_utils, "log_rerun_data", Mock())
    ticks = 0

    class EndTeleoperationError(Exception):
        pass

    def tick(_seconds):
        nonlocal ticks
        ticks += 1
        if ticks == 50:
            raise EndTeleoperationError

    monkeypatch.setattr("time.perf_counter", lambda: ticks / 50)
    monkeypatch.setattr(robot_utils, "precise_sleep", tick)
    script = Path(__file__).parents[2] / "examples/alohamini/teleoperate_bi.py"
    monkeypatch.setattr("sys.argv", [str(script), *options])

    with pytest.raises(EndTeleoperationError):
        runpy.run_path(str(script), run_name="__main__")

    config = make_client.call_args.args[0]
    assert config.observation_request_window == 3
    robot.connect.assert_called_once_with(include_cameras=False)
    robot.prime_observation_request_window.assert_called_once_with(include_cameras=False)
    assert robot.send_action.call_count == 50
    assert sum(call.kwargs["include_cameras"] for call in robot.get_observation.call_args_list) == camera_fps

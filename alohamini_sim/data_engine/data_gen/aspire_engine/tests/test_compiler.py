"""Unit tests for ASPIRE high-level command compilation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ASPIRE_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASPIRE_ENGINE_DIR))

from annotations import annotate
from task_compiler import compile_command, parse_natural_language


def test_compile_move_program() -> None:
    cmd = {
        "verb": "move",
        "object": "077_rubiks_cube",
        "goal": {"type": "zone", "xy": [-0.13, -0.35]},
    }

    program = compile_command(cmd)

    assert program == [
        {"skill": "goto_station", "target_xy": [-0.055, -0.45], "purpose": "pick"},
        {"skill": "pick", "object": "077_rubiks_cube", "pitch_deg": 60.0},
        {"skill": "carry", "to_station_purpose": "place"},
        {"skill": "place", "target_xy": [-0.13, -0.35]},
    ]


def test_compile_push_program() -> None:
    cmd = {
        "verb": "push",
        "object": "062_dice",
        "goal": {"type": "zone", "xy": [-0.2, -0.42]},
    }

    program = compile_command(cmd)

    assert program == [
        {"skill": "goto_station", "target_xy": [-0.13, -0.525], "purpose": "pick"},
        {"skill": "push", "object": "062_dice", "target_xy": [-0.2, -0.42]},
    ]


def test_compile_pick_program_and_annotations() -> None:
    cmd = {"verb": "pick", "object": "058_golf_ball", "goal": None}

    program = compile_command(cmd)
    labels = annotate(program, cmd)

    assert program == [
        {"skill": "goto_station", "target_xy": [-0.13, -0.375], "purpose": "pick"},
        {"skill": "pick", "object": "058_golf_ball", "pitch_deg": 60.0},
    ]
    assert labels["instruction"] == "Pick up the golf ball."
    assert labels["steps"] == ["go to the pick station", "pick up the golf ball"]


def test_compile_gather_offsets_three_objects() -> None:
    cmd = {
        "verb": "gather",
        "object": ["077_rubiks_cube", "062_dice", "058_golf_ball"],
        "goal": {"type": "zone", "xy": [-0.13, -0.35]},
    }

    program = compile_command(cmd)
    place_targets = [step["target_xy"] for step in program if step["skill"] == "place"]

    assert len(program) == 12
    assert place_targets == [[-0.13, -0.35], [-0.07, -0.35], [-0.19, -0.35]]
    assert len({tuple(xy) for xy in place_targets}) == 3


def test_parse_natural_language_round_trips() -> None:
    known = ["077_rubiks_cube", "062_dice", "058_golf_ball"]

    move_cmd = parse_natural_language("Move the rubik's cube to zone -0.13 -0.35", known)
    push_cmd = parse_natural_language("Push the dice to the point -0.2 -0.42", known)
    gather_cmd = parse_natural_language(
        "Gather the rubik's cube and golf ball at zone -0.1 -0.3",
        known,
    )

    assert move_cmd is not None
    assert push_cmd is not None
    assert gather_cmd is not None
    assert compile_command(move_cmd)[-1] == {"skill": "place", "target_xy": [-0.13, -0.35]}
    assert compile_command(push_cmd)[-1] == {
        "skill": "push",
        "object": "062_dice",
        "target_xy": [-0.2, -0.42],
    }
    assert [step["object"] for step in compile_command(gather_cmd) if step["skill"] == "pick"] == [
        "077_rubiks_cube",
        "058_golf_ball",
    ]


def test_parse_natural_language_unparseable_returns_none() -> None:
    assert parse_natural_language("please do something useful", ["077_rubiks_cube"]) is None


@pytest.mark.parametrize(
    "cmd, match",
    [
        ({"verb": "spin", "object": "077_rubiks_cube", "goal": None}, "unknown verb"),
        ({"verb": "pick", "object": "not_a_ycb_object", "goal": None}, "unknown object"),
        ({"verb": "move", "object": "077_rubiks_cube", "goal": None}, "requires a zone"),
        (
            {"verb": "gather", "object": "077_rubiks_cube", "goal": {"type": "zone", "xy": [0, 0]}},
            "non-empty list",
        ),
    ],
)
def test_invalid_commands_raise(cmd: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compile_command(cmd)

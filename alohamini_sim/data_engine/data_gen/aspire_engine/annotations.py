"""Deterministic language annotations for atomic skill programs."""

from __future__ import annotations

import re
from typing import Any


def annotate(program: list[dict[str, Any]], cmd: dict[str, Any]) -> dict[str, Any]:
    """Return episode-level and step-level language annotations."""

    return {
        "instruction": _instruction(cmd),
        "steps": [_step_text(step, cmd) for step in program],
    }


def _instruction(cmd: dict[str, Any]) -> str:
    verb = str(cmd.get("verb", "")).lower()
    objects = cmd.get("object")
    goal = cmd.get("goal")

    if verb == "move" and isinstance(objects, str):
        return f"Move the {_humanize_object(objects)} to {_goal_text(goal)}."
    if verb == "push" and isinstance(objects, str):
        return f"Push the {_humanize_object(objects)} to {_goal_text(goal)}."
    if verb == "pick" and isinstance(objects, str):
        return f"Pick up the {_humanize_object(objects)}."
    if verb == "gather" and isinstance(objects, list):
        return f"Gather {_join_objects(objects)} at {_goal_text(goal)}."
    return "Complete the requested object manipulation task."


def _step_text(step: dict[str, Any], cmd: dict[str, Any]) -> str:
    skill = step.get("skill")
    if skill == "goto_station":
        purpose = str(step.get("purpose", "pick"))
        if cmd.get("verb") == "push" and purpose == "pick":
            purpose = "push"
        return f"go to the {purpose} station"
    if skill == "pick":
        return f"pick up the {_humanize_object(str(step.get('object', 'object')))}"
    if skill == "carry":
        purpose = str(step.get("to_station_purpose", "place"))
        return f"carry it to the {purpose} station"
    if skill == "place":
        return f"place it at {_xy_text(step.get('target_xy'))}"
    if skill == "push":
        return (
            f"push the {_humanize_object(str(step.get('object', 'object')))} "
            f"to {_xy_text(step.get('target_xy'))}"
        )
    return f"run {skill}"


def _goal_text(goal: Any) -> str:
    if isinstance(goal, dict) and goal.get("type") == "zone":
        return f"the target zone at {_xy_text(goal.get('xy'))}"
    if isinstance(goal, dict) and goal.get("type") == "near_object":
        return f"near the {_humanize_object(str(goal.get('object', 'object')))}"
    return "the target area"


def _xy_text(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"({_format_number(value[0])}, {_format_number(value[1])})"
    return "the target point"


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:.3f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _join_objects(objects: list[Any]) -> str:
    names = [f"the {_humanize_object(str(obj))}" for obj in objects]
    if not names:
        return "the objects"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _humanize_object(object_id: str) -> str:
    name = re.sub(r"^\d{3}(?:-[a-z])?_", "", object_id)
    name = name.replace("_", " ").replace("-", " ")
    name = name.replace("rubiks", "rubik's")
    return name.strip() or object_id

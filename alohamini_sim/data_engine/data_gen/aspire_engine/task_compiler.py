"""Compile high-level language commands into atomic skill programs."""

from __future__ import annotations

import math
import re
from typing import Any

Command = dict[str, Any]
Program = list[dict[str, Any]]

PICK_PITCH_DEG = 60.0
PICK_STATION_XY = [-0.13, -0.45]
_TABLE_CENTER_XY = (-0.13, -0.45)
_VALID_VERBS = {"move", "gather", "push", "pick"}

# Keep this module independent from ManiSkill environment registration. The ids
# mirror the local YCB assets and common AlohaMini demo objects.
KNOWN_YCB_OBJECTS = {
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "022_windex_bottle",
    "024_bowl",
    "025_mug",
    "026_sponge",
    "028_skillet_lid",
    "029_plate",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "038_padlock",
    "040_large_marker",
    "042_adjustable_wrench",
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "048_hammer",
    "050_medium_clamp",
    "051_large_clamp",
    "052_extra_large_clamp",
    "053_mini_soccer_ball",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    "059_chain",
    "061_foam_brick",
    "062_dice",
    "063-a_marbles",
    "063-b_marbles",
    "065-a_cups",
    "065-b_cups",
    "065-c_cups",
    "065-d_cups",
    "065-e_cups",
    "065-f_cups",
    "065-g_cups",
    "065-h_cups",
    "065-i_cups",
    "065-j_cups",
    "070-a_colored_wood_blocks",
    "070-b_colored_wood_blocks",
    "071_nine_hole_peg_test",
    "072-a_toy_airplane",
    "072-b_toy_airplane",
    "072-c_toy_airplane",
    "072-d_toy_airplane",
    "072-e_toy_airplane",
    "073-a_lego_duplo",
    "073-b_lego_duplo",
    "073-c_lego_duplo",
    "073-d_lego_duplo",
    "073-e_lego_duplo",
    "073-f_lego_duplo",
    "073-g_lego_duplo",
    "077_rubiks_cube",
}

_DEFAULT_OBJECT_ORDER = [
    "065-a_cups",
    "077_rubiks_cube",
    "012_strawberry",
    "058_golf_ball",
    "062_dice",
    "009_gelatin_box",
    "070-a_colored_wood_blocks",
    "070-b_colored_wood_blocks",
    "073-f_lego_duplo",
    "061_foam_brick",
]

_SLOT_OFFSETS = [
    (0.0, 0.0),
    (0.075, 0.0),
    (-0.075, 0.0),
    (0.0, 0.075),
    (0.0, -0.075),
    (0.075, 0.075),
    (-0.075, 0.075),
    (0.075, -0.075),
    (-0.075, -0.075),
]

_PLACE_GRID_OFFSETS = [
    (0.0, 0.0),
    (0.06, 0.0),
    (-0.06, 0.0),
    (0.0, 0.06),
    (0.0, -0.06),
    (0.06, 0.06),
    (-0.06, 0.06),
    (0.06, -0.06),
    (-0.06, -0.06),
]


def compile_command(cmd: Command) -> Program:
    """Compile one command dictionary into an ordered atomic-skill program."""

    if not isinstance(cmd, dict):
        raise ValueError("command must be a JSON-like dictionary.")

    verb = str(cmd.get("verb", "")).strip().lower()
    if verb not in _VALID_VERBS:
        raise ValueError(f"unknown verb {verb!r}; expected one of {sorted(_VALID_VERBS)}.")

    goal = cmd.get("goal")
    if verb == "gather":
        objects = _validate_object_list(cmd.get("object"))
        target_xy = _validate_zone_goal(goal, verb)
        program: Program = []
        for index, object_id in enumerate(objects):
            place_xy = _offset_xy(target_xy, _PLACE_GRID_OFFSETS[index % len(_PLACE_GRID_OFFSETS)])
            program.extend(_move_program(object_id, place_xy))
        return program

    object_id = _validate_single_object(cmd.get("object"), verb)
    if verb == "move":
        return _move_program(object_id, _resolve_goal_xy(goal, verb))
    if verb == "push":
        target_xy = _resolve_goal_xy(goal, verb)
        return [
            {
                "skill": "goto_station",
                "target_xy": _object_station_xy(object_id),
                "purpose": "pick",
            },
            {"skill": "push", "object": object_id, "target_xy": target_xy},
        ]
    if verb == "pick":
        if goal is not None:
            _validate_goal(goal, verb)
        return _pick_program(object_id)

    raise ValueError(f"unsupported verb {verb!r}.")


def parse_natural_language(text: str, known_objects: list[str] | tuple[str, ...]) -> Command | None:
    """Parse a small set of deterministic natural-language command patterns."""

    if not text or not known_objects:
        return None

    normalized = _normalize_text(text)
    verb_match = re.search(r"\b(move|push|pick|gather|collect)\b", normalized)
    if verb_match is None:
        return None

    raw_verb = verb_match.group(1)
    verb = "gather" if raw_verb == "collect" else raw_verb
    known = [obj for obj in known_objects if _is_known_object(obj)]
    if not known:
        return None

    if verb == "pick":
        object_id = _find_first_object(normalized[verb_match.end() :], known)
        if object_id is None:
            return None
        return {"instruction": text, "verb": "pick", "object": object_id, "goal": None}

    goal = _parse_goal(normalized, known)
    if goal is None:
        return None

    object_text = _object_phrase(normalized, verb_match.end(), goal["_start"])
    if verb == "gather":
        objects = _find_all_objects(object_text, known)
        if len(objects) < 2:
            objects = _find_all_objects(normalized[: goal["_start"]], known)
        if len(objects) < 2:
            return None
        return {
            "instruction": text,
            "verb": "gather",
            "object": objects,
            "goal": _public_goal(goal),
        }

    object_id = _find_first_object(object_text, known)
    if object_id is None:
        object_id = _find_first_object(normalized[: goal["_start"]], known)
    if object_id is None:
        return None
    return {
        "instruction": text,
        "verb": verb,
        "object": object_id,
        "goal": _public_goal(goal),
    }


def _move_program(object_id: str, target_xy: list[float]) -> Program:
    return [
        {
            "skill": "goto_station",
            "target_xy": _object_station_xy(object_id),
            "purpose": "pick",
        },
        {"skill": "pick", "object": object_id, "pitch_deg": PICK_PITCH_DEG},
        {"skill": "carry", "to_station_purpose": "place"},
        {"skill": "place", "target_xy": target_xy},
    ]


def _pick_program(object_id: str) -> Program:
    return [
        {
            "skill": "goto_station",
            "target_xy": _object_station_xy(object_id),
            "purpose": "pick",
        },
        {"skill": "pick", "object": object_id, "pitch_deg": PICK_PITCH_DEG},
    ]


def _validate_single_object(value: Any, verb: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{verb!r} command requires object to be a YCB object id string.")
    if not _is_known_object(value):
        raise ValueError(f"unknown object {value!r}; expected a known YCB object id.")
    return value


def _validate_object_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("'gather' command requires object to be a non-empty list of YCB ids.")

    objects: list[str] = []
    for object_id in value:
        objects.append(_validate_single_object(object_id, "gather"))
    return objects


def _is_known_object(value: Any) -> bool:
    return isinstance(value, str) and value in KNOWN_YCB_OBJECTS


def _validate_zone_goal(goal: Any, verb: str) -> list[float]:
    if not isinstance(goal, dict) or goal.get("type") != "zone":
        raise ValueError(f"{verb!r} command requires goal {{'type': 'zone', 'xy': [x, y]}}.")
    return _xy(goal.get("xy"), f"{verb!r} goal.xy")


def _validate_goal(goal: Any, verb: str) -> None:
    if not isinstance(goal, dict):
        raise ValueError(f"{verb!r} goal must be a dictionary or null.")
    goal_type = goal.get("type")
    if goal_type == "zone":
        _xy(goal.get("xy"), f"{verb!r} goal.xy")
        return
    if goal_type == "near_object":
        _validate_single_object(goal.get("object"), verb)
        return
    raise ValueError(f"{verb!r} goal type must be 'zone' or 'near_object'.")


def _resolve_goal_xy(goal: Any, verb: str) -> list[float]:
    if not isinstance(goal, dict):
        raise ValueError(f"{verb!r} command requires a zone or near_object goal.")

    goal_type = goal.get("type")
    if goal_type == "zone":
        return _xy(goal.get("xy"), f"{verb!r} goal.xy")
    if goal_type == "near_object":
        object_id = _validate_single_object(goal.get("object"), verb)
        return _offset_xy(_object_station_xy(object_id), (0.06, 0.0))
    raise ValueError(f"{verb!r} goal type must be 'zone' or 'near_object'.")


def _xy(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two numeric coordinates.")
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric coordinates.") from exc


def _object_station_xy(object_id: str) -> list[float]:
    if object_id in _DEFAULT_OBJECT_ORDER:
        index = _DEFAULT_OBJECT_ORDER.index(object_id)
    else:
        index = sorted(KNOWN_YCB_OBJECTS).index(object_id) if object_id in KNOWN_YCB_OBJECTS else 0
    if index < len(_SLOT_OFFSETS):
        dx, dy = _SLOT_OFFSETS[index]
    else:
        ring = 1 + index // len(_SLOT_OFFSETS)
        angle_index = index % len(_SLOT_OFFSETS)
        angle = angle_index * (2.0 * 3.141592653589793 / len(_SLOT_OFFSETS))
        dx = 0.06 * ring * math.cos(angle)
        dy = 0.06 * ring * math.sin(angle)
    return [round(_TABLE_CENTER_XY[0] + dx, 6), round(_TABLE_CENTER_XY[1] + dy, 6)]


def _offset_xy(base_xy: list[float], offset: tuple[float, float]) -> list[float]:
    return [round(float(base_xy[0]) + offset[0], 6), round(float(base_xy[1]) + offset[1], 6)]


def _normalize_text(text: str) -> str:
    normalized = text.lower().replace("’", "'")
    normalized = re.sub(r"[\[\](),;:]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _parse_goal(text: str, known_objects: list[str]) -> dict[str, Any] | None:
    near_match = re.search(r"\b(?:near|next to|beside)\b", text)
    if near_match is not None:
        object_id = _find_first_object(text[near_match.end() :], known_objects)
        if object_id is not None:
            return {"type": "near_object", "object": object_id, "_start": near_match.start()}

    coord_match = re.search(
        r"\b(?:to|at|in|into)\b\s+(?:the\s+)?"
        r"(?:(?:target|goal)\s+)?(?:zone|region|area|point|location)?\s*"
        r"(?:at\s+)?"
        r"([-+]?(?:\d+\.\d+|\d+|\.\d+))\s*,?\s+"
        r"([-+]?(?:\d+\.\d+|\d+|\.\d+))",
        text,
    )
    if coord_match is not None:
        return {
            "type": "zone",
            "xy": [float(coord_match.group(1)), float(coord_match.group(2))],
            "_start": coord_match.start(),
        }

    to_match = re.search(r"\bto\b", text)
    if to_match is not None:
        object_id = _find_first_object(text[to_match.end() :], known_objects)
        if object_id is not None:
            return {"type": "near_object", "object": object_id, "_start": to_match.start()}
    return None


def _public_goal(goal: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in goal.items() if not key.startswith("_")}


def _object_phrase(text: str, start: int, stop: int) -> str:
    phrase = text[start:stop]
    phrase = re.sub(r"\b(the|a|an|object|objects|item|items)\b", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase.strip()


def _find_first_object(text: str, known_objects: list[str]) -> str | None:
    matches = _find_object_matches(text, known_objects)
    return matches[0][1] if matches else None


def _find_all_objects(text: str, known_objects: list[str]) -> list[str]:
    found: list[str] = []
    for _, object_id in _find_object_matches(text, known_objects):
        if object_id not in found:
            found.append(object_id)
    return found


def _find_object_matches(text: str, known_objects: list[str]) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for object_id in known_objects:
        for alias in _aliases_for_object(object_id):
            match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text)
            if match is not None:
                matches.append((match.start(), object_id))
                break
    return sorted(matches, key=lambda item: item[0])


def _aliases_for_object(object_id: str) -> list[str]:
    name = re.sub(r"^\d{3}(?:-[a-z])?_", "", object_id)
    words = name.replace("_", " ").replace("-", " ")
    aliases = {
        object_id,
        object_id.replace("_", " "),
        words,
        words.replace("rubiks", "rubik's"),
        words.replace("rubiks", "rubiks"),
    }
    if words.endswith("s"):
        aliases.add(words[:-1])
    return sorted(aliases, key=len, reverse=True)

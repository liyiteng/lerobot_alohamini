"""Loader for the ASPIRE-style skill library.

Each JSON entry stores {failure_signature, when_to_apply, repair_strategy, code_sketch}
— compact in-context guidance (ASPIRE §2.2), not full task programs. `match()` does a
naive keyword scan over failure signatures so a coding agent (or a human) can retrieve
candidate repairs from observed symptoms/trace lines.
"""
from __future__ import annotations

import json
from pathlib import Path

_LIB_DIR = Path(__file__).parent


def load_skills() -> dict[str, dict]:
    skills = {}
    for p in sorted(_LIB_DIR.glob("*.json")):
        with open(p, encoding="utf-8") as fh:
            entry = json.load(fh)
        skills[entry["name"]] = entry
    return skills


def match(symptoms: str, top_k: int = 3) -> list[dict]:
    """Rank skills by keyword overlap between `symptoms` and each failure signature."""
    words = {w.lower().strip(".,:;()[]'\"") for w in symptoms.split() if len(w) > 3}
    scored = []
    for entry in load_skills().values():
        sig = entry.get("failure_signature", {})
        hay = " ".join(sig.get("symptoms", []) + sig.get("trace_evidence", [])).lower()
        score = sum(1 for w in words if w in hay)
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda t: -t[0])
    return [e for _, e in scored[:top_k]]


if __name__ == "__main__":
    lib = load_skills()
    print(f"{len(lib)} skills loaded:")
    for name, e in lib.items():
        print(f"  - {name:32s} [{e.get('category','?')}]")
    print("\nmatch('object pushed away during descent, palm contact'):")
    for e in match("object pushed away during descent, palm contact impulse"):
        print(f"  -> {e['name']}: {e['repair_strategy'][:90]}...")

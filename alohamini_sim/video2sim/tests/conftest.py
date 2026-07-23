"""Pytest path setup: make the ``video2sim`` package importable from the repo root.

Run the suite with the MAIN venv interpreter (the only one with torch, gsplat,
pycolmap, open3d and scipy installed):

    cd /home/perelman/AlohaMini/video2sim
    /home/perelman/Basic_RL/.venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

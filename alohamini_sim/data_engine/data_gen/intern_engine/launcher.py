"""Command-line launcher for the AlohaMini ManiSkill data engine."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import load_config
from .pipeline import ManiSkillDataEngine


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    summary = ManiSkillDataEngine(cfg).run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""``python -m video2sim`` entry point — delegates to :mod:`video2sim.cli`.

Usage:
  python -m video2sim run VIDEO.mp4 --workdir OUT [--config cfg.yaml]
      [--from STAGE] [--until STAGE] [--gui] [--force]
"""

from __future__ import annotations

import sys

from video2sim.cli import main

if __name__ == "__main__":
    sys.exit(main())

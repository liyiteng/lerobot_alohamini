#!/usr/bin/env python3

"""Record AlohaMini datasets with 50 Hz control and fresh camera frames."""

try:
    from . import record_bi
    from .record_utils_multirate import record_loop
except ImportError:  # Direct execution from the repository root.
    import record_bi
    from record_utils_multirate import record_loop


if __name__ == "__main__":
    record_bi.record_loop = record_loop
    record_bi.main()

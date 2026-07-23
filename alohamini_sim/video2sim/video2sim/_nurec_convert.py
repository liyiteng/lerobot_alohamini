"""3DGS PLY -> NuRec USDZ conversion helper (runs under the NuRec venv ONLY).

Converts a standard 3DGS PLY to NuRec USDZ for Isaac Sim 5.x, without the
3dgrut CUDA build: PLYImporter -> AttributesExportAdapter -> NuRecExporter.

Pipeline stage: pruned SH3 PLY (from ``video2sim.export``) -> THIS -> USDZ
for Isaac Sim scene assembly. Invoked by ``video2sim.export`` as a
subprocess; can also be run directly:

  /home/perelman/nurec-venv/bin/python .../video2sim/_nurec_convert.py in.ply out.usdz

Interpreter contract (documented, not auto-switched): this script requires
/home/perelman/nurec-venv/bin/python (numpy/torch/pxr/msgpack/ncore); it will
NOT import under the main Basic_RL venv.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import List, Optional

TAG = "[nurec]"


def _install_threedgrut_stubs() -> None:
    """Stub the heavy threedgrut submodules before any threedgrut import.

    threedgrut/__init__ pulls in the interactive GUI stack, and sh_bake pulls
    the full datasets stack (cv2/kornia/kaolin/...) just for one dataloader
    helper; stub both — export only needs numpy/torch/pxr/msgpack/ncore.
    """
    _gui = types.ModuleType("threedgrut.gui")
    sys.modules["threedgrut.gui"] = _gui
    _ds = types.ModuleType("threedgrut.datasets")
    _dsu = types.ModuleType("threedgrut.datasets.utils")
    _dsu.configure_dataloader_for_platform = lambda *a, **k: (a[0] if a else None)
    _dsu.DEFAULT_DEVICE = "cpu"
    _ds.utils = _dsu
    sys.modules["threedgrut.datasets"] = _ds
    sys.modules["threedgrut.datasets.utils"] = _dsu


def convert(src: Path, dst: Path, threedgrut_root: Path) -> Path:
    """Load ``src`` PLY and export it as a NuRec USDZ at ``dst``."""
    sys.path.insert(0, str(threedgrut_root))
    _install_threedgrut_stubs()
    # imports must come AFTER the stubs are registered in sys.modules
    from threedgrut.export.importers.ply import PLYImporter            # noqa: E402
    from threedgrut.export.adapter import AttributesExportAdapter      # noqa: E402
    from threedgrut.export.usd.nurec.exporter import NuRecExporter     # noqa: E402

    attrs, caps = PLYImporter().load(str(src))
    model = AttributesExportAdapter(attrs, caps, is_preactivation=True, device="cpu")
    print(f"{TAG} loaded {src.name}: n={model.get_positions().shape[0]}", flush=True)
    exporter = NuRecExporter(export_cameras=False, export_post_processing=False)
    exporter.export(model, dst, dataset=None, conf=None)
    print(f"{TAG} wrote {dst}", flush=True)
    return dst


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="_nurec_convert.py",
        description="standard 3DGS PLY -> NuRec USDZ (NuRec venv only)")
    ap.add_argument("src", type=Path, help="input 3DGS PLY (62-prop, SH3-padded)")
    ap.add_argument("dst", type=Path, help="output NuRec USDZ")
    ap.add_argument("--threedgrut-root", type=Path,
                    default=Path("/home/perelman/3dgrut"),
                    help="3dgrut checkout providing the threedgrut export modules")
    args = ap.parse_args(argv)
    convert(args.src, args.dst, args.threedgrut_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

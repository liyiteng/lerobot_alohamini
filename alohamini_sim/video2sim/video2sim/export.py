"""Trained splat -> NuRec USDZ export (SH3 PLY, floater pruning, USDZ).

What: turns the trained gsplat checkpoint (``splat.pt``) into an Isaac Sim
5.x NuRec asset in three steps:

  1. ``splat.pt`` -> standard 62-prop 3DGS PLY with ZERO-padded SH degree-3
     rest bands. The NuRec runtime config assumes radiance_sph_degree=3 and
     an SH0-only payload sends it down a pathological slow path
     (~47 s/frame); zero f_rest is visually identical to SH0.
  2. Free-space floater pruning: the MV-consistent TSDF cloud (from
     ``video2sim.fuse``) is the surface oracle — gaussians farther than
     --radius (default 0.20 m) from ANY surface point are drifting fog and
     get dropped before the NuRec export. Conservative on purpose: TSDF
     holes (unobserved areas) only lose gaussians beyond the radius.
     SPATIAL criterion ONLY — opacity pruning is FORBIDDEN for MCMC-trained
     ensembles (min_opacity pruning deletes ~79% of an MCMC ensemble, which
     keeps many meaningful low-opacity gaussians); --min-opacity is exposed
     for non-MCMC splats but defaults to 0.0 (off).
  3. Pruned PLY -> NuRec USDZ via subprocess under the NuRec venv, running
     ``video2sim/_nurec_convert.py`` (PLYImporter -> AttributesExportAdapter
     -> NuRecExporter, no 3dgrut CUDA build needed).

Pipeline stage: gsplat training output (+ fuse TSDF cloud) -> THIS (export)
-> scene.usdz for Isaac Sim scene assembly.

Interpreter contract (documented, not auto-switched): steps 1-2 run under
the main venv /home/perelman/Basic_RL/.venv/bin/python (torch/numpy/scipy);
step 3 is executed as a subprocess under /home/perelman/nurec-venv/bin/python.

Usage:
  python -m video2sim.export [--workdir W] [--splat W/splat.pt]
      [--tsdf W/fuse/mvtsdf.ply] [--radius 0.20] [--min-opacity 0.0]
      [--out-ply P] [--pruned-ply P] [--usdz W/export/scene.usdz]
      [--no-prune] [--no-usdz]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

TAG = "[export]"

C0 = 0.28209479177387814  # SH DC basis; standard 3DGS: color = 0.5 + C0 * f_dc

NUREC_PYTHON = Path("/home/perelman/nurec-venv/bin/python")
THREEDGRUT_ROOT = Path("/home/perelman/3dgrut")


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1: splat.pt -> standard 3DGS PLY (62 props, SH3 zero-padded)
# ---------------------------------------------------------------------------

def export_sh3_ply(splat_pt: Path, out_ply: Path) -> int:
    """Export a gsplat ``splat.pt`` to the standard 3DGS PLY layout.

    Attributes: x,y,z, nx,ny,nz(0), f_dc_0..2, f_rest_0..44(0),
    opacity(logit), scale_0..2(log), rot_0..3 (wxyz, normalized).
    Returns the number of gaussians written.
    """
    sp = torch.load(str(splat_pt), map_location="cpu", weights_only=False)
    means = sp["means"].numpy().astype(np.float32)
    quats = sp["quats"].numpy().astype(np.float32)
    quats = quats / np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8, None)
    log_scales = sp["scales"].numpy().astype(np.float32)
    logit_op = sp["opacities"].numpy().astype(np.float32)
    cols = sp["colors"]
    if cols.max() > 1.5 or cols.min() < -0.5:
        cols = torch.sigmoid(cols)
    rgb = cols.numpy().astype(np.float32)
    f_dc = (rgb - 0.5) / C0        # standard 3DGS: color = 0.5 + C0 * f_dc

    N = len(means)
    # pad zero SH rest bands (degree 3): the NuRec runtime config assumes
    # radiance_sph_degree=3 and an SH0-only payload sends it down a pathological
    # slow path (~47s/frame); zero f_rest is visually identical to SH0
    fields = ["x", "y", "z", "nx", "ny", "nz",
              "f_dc_0", "f_dc_1", "f_dc_2"] + \
        [f"f_rest_{i}" for i in range(45)] + ["opacity",
              "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([
        means, np.zeros((N, 3), np.float32), f_dc,
        np.zeros((N, 45), np.float32), logit_op[:, None],
        log_scales, quats], axis=1).astype(np.float32)

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ply, "wb") as f:
        hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {N}"]
        hdr += [f"property float {n}" for n in fields]
        hdr += ["end_header"]
        f.write(("\n".join(hdr) + "\n").encode())
        f.write(data.tobytes())
    _log(f"wrote {out_ply}: {N} gaussians, {len(fields)} props")
    return N


# ---------------------------------------------------------------------------
# Step 2: free-space floater pruning against the TSDF surface oracle
# ---------------------------------------------------------------------------

def read_ply(path: Path) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Read a binary-little-endian PLY into a structured array + prop list."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        n, props = 0, []
        while True:
            ln = f.readline()
            if ln.startswith(b"element vertex"):
                n = int(ln.split()[2])
            elif ln.startswith(b"property"):
                t, nm = ln.split()[1:3]
                props.append((nm.decode(),
                              {"float": "<f4", "double": "<f8", "uchar": "u1"}[t.decode()]))
            elif ln.startswith(b"end_header"):
                break
        arr = np.frombuffer(f.read(), dtype=props, count=n)
    return arr, props


def prune_floaters(
    splat_ply: Path,
    tsdf_ply: Path,
    out_ply: Path,
    radius: float = 0.20,
    min_opacity: float = 0.0,
) -> int:
    """Drop gaussians farther than ``radius`` from any TSDF surface point.

    SPATIAL criterion only. ``min_opacity`` MUST stay 0.0 for MCMC-trained
    splats — opacity pruning deletes the low-opacity bulk of an MCMC
    ensemble (~79% loss observed). Returns the number of kept gaussians.
    """
    splat, sprops = read_ply(splat_ply)
    tsdf, _ = read_ply(tsdf_ply)
    P = np.stack([splat["x"], splat["y"], splat["z"]], 1)
    S = np.stack([tsdf["x"], tsdf["y"], tsdf["z"]], 1).astype(np.float32)
    tree = cKDTree(S)
    d, _ = tree.query(P, k=1, workers=-1)
    op = 1.0 / (1.0 + np.exp(-splat["opacity"]))
    keep = (d <= radius) & (op >= min_opacity)
    _log(f"{len(P)} gaussians -> keep {keep.sum()} "
         f"(free-space {int((d > radius).sum())}, dust {int((op < min_opacity).sum())})")
    out = splat[keep]
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {len(out)}"]
    hdr += [f"property {'float' if dt == '<f4' else 'uchar'} {nm}" for nm, dt in sprops]
    hdr += ["end_header"]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ply, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode())
        f.write(out.tobytes())
    _log(f"wrote {out_ply}")
    return int(keep.sum())


# ---------------------------------------------------------------------------
# Step 3: PLY -> NuRec USDZ (subprocess under the NuRec venv)
# ---------------------------------------------------------------------------

def convert_to_usdz(
    ply: Path,
    usdz: Path,
    nurec_python: Path = NUREC_PYTHON,
    threedgrut_root: Path = THREEDGRUT_ROOT,
) -> Path:
    """Run video2sim/_nurec_convert.py under the NuRec venv; return usdz path."""
    if not nurec_python.exists():
        raise SystemExit(f"{TAG} ERROR: NuRec interpreter not found: {nurec_python}")
    script = Path(__file__).with_name("_nurec_convert.py")
    usdz.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(nurec_python), str(script), str(ply), str(usdz),
           "--threedgrut-root", str(threedgrut_root)]
    _log("nurec convert: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    _log(f"usdz done: {usdz}")
    return usdz


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_export(
    splat_pt: Path,
    out_ply: Path,
    usdz: Path,
    tsdf: Optional[Path] = None,
    pruned_ply: Optional[Path] = None,
    radius: float = 0.20,
    min_opacity: float = 0.0,
    do_prune: bool = True,
    do_usdz: bool = True,
    nurec_python: Path = NUREC_PYTHON,
    threedgrut_root: Path = THREEDGRUT_ROOT,
) -> Path:
    """splat.pt -> SH3 PLY [-> pruned PLY] [-> NuRec USDZ]. Returns last output."""
    _log(f"step 1/3: {splat_pt} -> {out_ply}")
    export_sh3_ply(splat_pt, out_ply)

    final_ply = out_ply
    if do_prune:
        if tsdf is None:
            raise SystemExit(f"{TAG} ERROR: pruning requires --tsdf (or pass --no-prune)")
        if pruned_ply is None:
            pruned_ply = out_ply.with_name(out_ply.stem + "_pruned.ply")
        _log(f"step 2/3: prune vs {tsdf} (radius {radius}, min_opacity {min_opacity})")
        prune_floaters(out_ply, tsdf, pruned_ply, radius=radius, min_opacity=min_opacity)
        final_ply = pruned_ply
    else:
        _log("step 2/3: pruning skipped (--no-prune)")

    if do_usdz:
        _log(f"step 3/3: {final_ply} -> {usdz}")
        return convert_to_usdz(final_ply, usdz,
                               nurec_python=nurec_python,
                               threedgrut_root=threedgrut_root)
    _log("step 3/3: usdz skipped (--no-usdz)")
    return final_ply


def _default_tsdf(workdir: Path) -> Path:
    """TSDF cloud where fuse writes it (WORKDIR/fuse/mvtsdf.ply, or WORKDIR root)."""
    for cand in (workdir / "fuse" / "mvtsdf.ply", workdir / "mvtsdf.ply"):
        if cand.exists():
            return cand
    raise SystemExit(
        f"{TAG} ERROR: no TSDF cloud at {workdir / 'fuse' / 'mvtsdf.ply'} or "
        f"{workdir / 'mvtsdf.ply'}; pass --tsdf, or --no-prune to skip pruning")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m video2sim.export",
        description="splat.pt -> SH3-padded 3DGS PLY -> floater prune -> NuRec USDZ")
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="pipeline working dir; default root for --splat/--tsdf/outputs")
    ap.add_argument("--splat", type=Path, default=None,
                    help="trained gsplat checkpoint (default: WORKDIR/splat.pt)")
    ap.add_argument("--tsdf", type=Path, default=None,
                    help="MV-consistent TSDF cloud PLY, the surface oracle for pruning "
                         "(default: WORKDIR/fuse/mvtsdf.ply, else WORKDIR/mvtsdf.ply)")
    ap.add_argument("--radius", type=float, default=0.20,
                    help="max distance to nearest surface point (m)")
    ap.add_argument("--min-opacity", type=float, default=0.0,
                    help="also drop near-transparent dust regardless of position. "
                         "MUST stay 0.0 for MCMC-trained splats (opacity pruning "
                         "deletes ~79%% of an MCMC ensemble)")
    ap.add_argument("--out-ply", type=Path, default=None,
                    help="SH3-padded PLY output (default: WORKDIR/export/splat_sh3.ply)")
    ap.add_argument("--pruned-ply", type=Path, default=None,
                    help="pruned PLY output (default: <out-ply stem>_pruned.ply)")
    ap.add_argument("--usdz", type=Path, default=None,
                    help="NuRec USDZ output (default: WORKDIR/export/scene.usdz)")
    ap.add_argument("--no-prune", action="store_true",
                    help="skip floater pruning (convert the unpruned PLY)")
    ap.add_argument("--no-usdz", action="store_true",
                    help="stop after the PLY step(s); do not run the NuRec venv")
    ap.add_argument("--nurec-python", type=Path, default=NUREC_PYTHON,
                    help="NuRec venv interpreter for the USDZ step")
    ap.add_argument("--threedgrut-root", type=Path, default=THREEDGRUT_ROOT,
                    help="3dgrut checkout providing the threedgrut export modules")
    args = ap.parse_args(argv)

    splat = args.splat if args.splat is not None else args.workdir / "splat.pt"
    out_ply = (args.out_ply if args.out_ply is not None
               else args.workdir / "export" / "splat_sh3.ply")
    usdz = args.usdz if args.usdz is not None else args.workdir / "export" / "scene.usdz"
    tsdf = args.tsdf
    if tsdf is None and not args.no_prune:
        tsdf = _default_tsdf(args.workdir)

    run_export(splat_pt=splat, out_ply=out_ply, usdz=usdz, tsdf=tsdf,
               pruned_ply=args.pruned_ply, radius=args.radius,
               min_opacity=args.min_opacity, do_prune=not args.no_prune,
               do_usdz=not args.no_usdz, nurec_python=args.nurec_python,
               threedgrut_root=args.threedgrut_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

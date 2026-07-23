# video2sim

**Phone video → photoreal Isaac Sim scene with a physics collider, on an 8 GB GPU.**

video2sim turns a single hand-held phone walkthrough into an Isaac Sim 5.x stage:
a NuRec Gaussian-splat room you can render photorealistically, plus an invisible,
metric, gravity-aligned TSDF collider the robot can actually touch.

It is **SfM-free**: no COLMAP/GLOMAP mapping in the loop. Camera poses come from
LingBot-Map streaming inference, get *jointly refined during splat training*
(per-view se3 through the rasterizer gradients, view 0 anchored), and the splat is
initialized **densely from the MV-consistent TSDF cloud** — InstantSplat-style.
That dense init is what makes the whole thing fast: **10k iterations ≈ 20 min**
end-to-end training instead of the 80k-iteration growth schedule it replaces.

Every default in this package is a hard-won fix from ~10 failed runs on the
validated room3 capture. The defaults of record live in
[`configs/default.yaml`](configs/default.yaml) — do **not** "improve" them
without re-validating.

---

## Architecture

Each stage is one module: importable *and* runnable as `python -m video2sim.<stage>`.
The orchestrator (`python -m video2sim run`) sequences them as subprocesses, routing
each stage to the interpreter its contract demands (it never auto-switches inside a
process).

```
 phone video (.mp4)
      |
      v
 [extract]        video -> frames/f_%04d.jpg  (ffmpeg, fps 8)      | main venv (Basic_RL)
      |
      v
 [lingbot_infer]  frames -> lingbot/pred-*.pt                      | main venv
      |           streaming poses + depth; fp8 KV + sliding        |  + PYTHONPATH=<lingbot fork>
      |           window 48 (the 8 GB recipe)                      |  + ninja on PATH
      +---------------------------+
      |                           |
      v                           v
 [fuse]                       [to_colmap]                          | main venv
  MV-consistency filter +      LingBot E/K + TSDF cloud            |
  4 mm TSDF fusion             -> COLMAP text sparse/0             |
  -> fuse/mvtsdf.ply(+mesh)    (dense 350k init, NO SfM)           |
      |                           |
      v                           v
 [refine]                     [train]                              | main venv, GPU EXCLUSIVE
  island filter, RANSAC        gsplat MCMC 10k iters,              |
  floor -> z=0, metric         packed=True, D-SSIM, PPISP-lite,    |
  scale (ceiling 2.4 m)        joint pose-opt (view 0 anchor)      |
  -> final_mesh/cloud/align    -> splat.pt (+refined_poses.npz)    |
      |                           |
      v                           v
 [scene_prep]                 [export]                             | main venv;
  decimate to 300k tris        splat.pt -> SH3 zero-padded PLY     |  USDZ substep runs under
  -> collider.npz              -> free-space prune (TSDF oracle)   |  nurec-venv (subprocess)
      |                        -> export/scene.usdz                |
      +------------+--------------+
                   v
              [scene]   splat (align transform) + invisible        | isaac5-venv
                        collider + robot + dome 450
                        -> scene/scene.usd            [--gui]

 side tools:  [spots]  open-spot finder + capture-camera pick      | main venv
              [check_env]  preflight pass/fail table               | any python (probes venvs)
```

Interpreter contract (documented in every module docstring, never auto-switched):

| Stage(s) | Interpreter |
|---|---|
| extract, lingbot_infer, fuse, refine, to_colmap, train, export (steps 1–2), scene_prep, spots | `/home/perelman/Basic_RL/.venv/bin/python` |
| export step 3 (PLY → NuRec USDZ, `_nurec_convert.py`) | `/home/perelman/nurec-venv/bin/python` |
| scene (Isaac USD assembly) | `/home/perelman/isaac5-venv/bin/python` |
| LingBot child process | main venv **+** `PYTHONPATH=<lingbot fork>` **+** `ninja` on PATH (the wrapper sets both) |

---

## Requirements

- **GPU**: NVIDIA, **8 GiB VRAM minimum** (validated card: RTX 4060 8 GB — the whole
  recipe is 8 GB-bounded). The GPU must be **exclusive** during LingBot inference,
  CUDA SfM and splat training: an open Isaac GUI alone eats 3–4 GB and OOMs the run.
- **RAM**: **32 GB** (validated on a 31 GB box; the trainer caches images as uint8
  ≈ 3 GB — float32 would be ~12 GB — and fuse warns at 45 GB RSS).
- **Binaries**: `ffmpeg` / `ffprobe` on PATH.
- **Venv inventory**:

| Venv | Path | Must provide |
|---|---|---|
| main | `/home/perelman/Basic_RL/.venv` | torch + CUDA, gsplat ≥ 1.5 (MCMCStrategy + packed rasterization), pycolmap, open3d, scipy, PIL, **ninja in `bin/`** (the fp8-KV CUDA extension is JIT-compiled) |
| NuRec | `/home/perelman/nurec-venv` | pxr (USD), torch, msgpack, ncore — for the threedgrut export path |
| Isaac | `/home/perelman/isaac5-venv` | isaacsim 5.x + pxr (incl. `omni.usd.schema.omni_nurec_types`) |

- **Checkouts / models**: LingBot-Map RTX4060-8GB fork (with `scripts/predict_stream.py`),
  LingBot checkpoint `lingbot-map-long.pt`, and a `3dgrut` checkout
  (`/home/perelman/3dgrut`) for the NuRec exporter modules.

## Setup

Run the preflight check first — it verifies every runtime the pipeline touches
*before* you burn an hour on a run (venv imports in subprocesses, ffmpeg, LingBot
fork + ckpt, ninja, GPU VRAM/exclusivity, RAM), and exits nonzero on any FAIL:

```bash
cd /home/perelman/AlohaMini/video2sim
python3 -m video2sim.check_env
```

`check_env` is stdlib-only and runs under any Python; heavy imports are probed in
subprocesses under their own interpreters. Read the WARN rows too — a "busy GPU"
warning means training/inference will OOM if you proceed.

## Quickstart

One command, end-to-end (idempotent — re-running skips stages whose outputs exist):

```bash
cd /home/perelman/AlohaMini/video2sim
/home/perelman/Basic_RL/.venv/bin/python -m video2sim run ~/room.mp4 --workdir runs/room1
```

Then open the assembled scene in the Isaac GUI (needs the GPU free again):

```bash
/home/perelman/Basic_RL/.venv/bin/python -m video2sim run --workdir runs/room1 --gui
```

Outputs land in the workdir: `frames/`, `lingbot/pred-*.pt`, `fuse/mvtsdf.ply(+_mesh)`,
`final_{mesh,cloud}.ply` + `final_align.npz`, `sparse/0/`, `splat.pt`,
`export/scene.usdz`, `collider.npz`, `scene/scene.usd`.

Per-stage overrides go through a YAML config (`--config`, key `stage_args`) or by
running any single stage directly, e.g.
`python -m video2sim.train --workdir runs/room1 --iters 20000 --video`.
For a ~60 s capture the wall-clock is ≈ 30–40 min, training-dominated (~20 min).

## The standard recipe (and WHY each number is what it is)

Mirrors `configs/default.yaml` == the argparse defaults. Every WHY below was measured
on this hardware; the odd-looking constants are failure post-mortems, not taste.

| Stage | Setting | WHY (measured) |
|---|---|---|
| extract | `fps=8`, `-q:v 2` | Dense enough for sequential matching at walking speed, sparse enough to stay tractable. Thinning happens HERE — downstream is **stride 1 always**. Near-lossless JPEG: feature extractors are quality-sensitive. |
| extract | autorotate, no transpose filter | ffmpeg applies rotation metadata before `-vf`; adding transpose double-rotates. Orientation judged on DISPLAY size. |
| (any SfM validation) | `--single-camera` **MANDATORY** | Per-frame independent cameras collapse focal lengths to −1e6…+8e4 → training loss oscillation + fog. Shared camera recovers fx 1642 (independently matching LingBot's 1643). |
| lingbot_infer | fp8 KV cache + sliding-window **48** | THE 8 GB recipe. sw64+fp8 = confirmed OOM (KV +0.6 GB collides with the attention peak). |
| lingbot_infer | `max_frame_num` = actual frames + 8 | FlashInfer pre-allocates the paged-KV pool from it; the fork's default 1024 was 6.7 GB on its own → OOM on 8 GB. |
| lingbot_infer | `image_size` stays 518 | The ckpt pos_embed (1,1370,1024) is size-bound — never override. |
| fuse | voxel 4 mm, trunc 2 cm, conf > 2.3, neighbors ±4/8/12, ≥ 2 consistent | Collider-grade surface that still fits RAM. Kept-pixel % doubles as a pose-health metric: same-track ~88 %, cross-track collapses to ~58 %. |
| refine | camera-track gates on floor RANSAC (above > 0.9, med > 0.2, spread < 0.45) | The biggest plane is often a WALL. The spread gate is what rejects walls: a hand-held phone keeps near-constant height over the FLOOR but varying distance to walls. |
| refine | ceiling anchor 2.4 m, fallback hand-height 1.35 m | Metric scale from floor→ceiling when a horizontal ceiling plane is visible; else phone hand height over the floor. |
| to_colmap | dense init **350k** points from the TSDF cloud | The InstantSplat fast path: growth-free training converged in **10k iters vs 80k**, with **38× fewer floaters**. LingBot extrinsics are world→cam in COLMAP convention — qvec/tvec written directly, no inversion. |
| train | `iters=10000`, `cap=350000`, `downscale=1` | 10k suffices *because* of the dense init. cap 500k = measured OOM; 350k @1080p fits. (Yesterday's 1080p OOM streak was actually the extra depth render channel, not the resolution.) |
| train | `packed=True` **ALWAYS** | Sparse tile-intersection buffers are the difference between OOM and full-res on 8 GB (unpacked died at ~260k gaussians even at 2× downscale). |
| train | lrs keyed **BY NAME** | `torch.nn.ParameterDict` sorts keys alphabetically; a positional zip silently shuffled every lr (opacities got 1e-3 instead of 5e-2) — the single biggest cause of the fog splats. |
| train | `scene_scale` capped at 6.0 | Corridor/forward-motion scenes give a huge camera spread that over-inflates means-lr + MCMC noise and collapses opacity to fog. |
| train | D-SSIM 0.2 term | Plain L1 is minimized just as well by big translucent blobs — the fog optimum. |
| train | joint pose-opt on, lr 1e-4, **view 0 anchored** | Absorbs feedforward pose drift without SfM: LingBot poses measured at **54 cm RMS** vs a GLOMAP/BA trajectory (Umeyama residual; reproduced 53.8). Corrections keep growing with budget (1 cm median @20k → 3.8 cm @80k) — joint opt asymptotes toward SfM. |
| train | PPISP-lite per-view exposure/WB | Phone auto-exposure/AWB otherwise gets baked into the splat as geometric mud; identity at novel-view time. |
| train | `--depth-reg` **SAME-track poses only** | LingBot depth + poses from a *different* track = fog hedging (every gaussian goes translucent); MV re-fusion drops 88 % → 58 %. Depth and poses are one body. |
| train | optimizer step first, THEN MCMC relocation | Official ordering; relocating first applied stale grads to teleported gaussians. |
| export | SH3 **zero-padded** PLY (62 props) | NuRec assumes `radiance_sph_degree=3`; an SH0-only payload hits a pathological **~47 s/frame** slow path. Zero `f_rest` is visually identical to SH0. |
| export | free-space prune, radius 0.20 m, **spatial criterion ONLY** | TSDF cloud is the surface oracle; conservative on purpose (TSDF holes only lose gaussians beyond the radius). **Opacity pruning is FORBIDDEN** for MCMC ensembles — `min_opacity` cuts deleted ~79 % of the splat (MCMC keeps meaningful low-opacity gaussians). |
| scene_prep | collider budget 300k tris | Full contact honesty without blowing up PhysX cooking (same budget as the CuRobo env-mesh recipe). |
| scene | dome intensity **450**, no key light | The mesh-room recipe (dome 1200 + key 2000) overexposes the NuRec splat to white blur; approved GUI views used ~350–450. |
| scene | collider identity transform; only the splat gets the align matrix | The refined mesh is already metric/aligned (floor z=0); the splat still lives in the raw LingBot frame. `Gf.Matrix4d` is row-vector (`p' = p @ M`). |
| spots | candidates within **1.2 m of the walked track** | The global max-clearance cell was reconstruction garbage OUTSIDE the walls — noise past the room boundary looks like open floor to a naive distance transform. |
| spots | demo camera = a **real capture-trajectory pose** | A synthetic camera 2.4 m out sat inside splat furniture and filmed only blur; a pose the videographer actually stood at is guaranteed fog-free. |

## Troubleshooting

Failure modes actually hit during development, symptom-first:

- **Splat renders as fog / translucent blobs.** Check in order: (1) lrs keyed by
  name, not position (`ParameterDict` alphabetical-sort trap); (2) D-SSIM term
  present; (3) `scene_scale` capped (corridor scenes); (4) you did NOT pair
  `--depth-reg` with poses from a different track.
- **Training OOM on 8 GB.** `packed=True` must be on; cap ≤ 350k (500k OOMs);
  drop `--depth-reg` (the extra `RGB+ED` render channel is what killed 1080p runs);
  make sure nothing else holds VRAM — the Isaac GUI alone eats 3–4 GB.
- **LingBot OOM in the FlashInfer paged-KV pool.** `max_frame_num` must be sized to
  the sequence (the wrapper does this; the fork default 1024 pre-allocates 6.7 GB).
  Sliding window > 48 with fp8 also OOMs. GPU must be exclusive.
- **`ninja: not found` during LingBot startup.** The fp8-KV CUDA extension is
  JIT-compiled and needs ninja in the *main venv bin* (leads PATH in the child);
  `check_env` verifies this.
- **NuRec renders at ~47 s/frame.** Your PLY is SH0-only. Export through
  `video2sim.export` — it zero-pads the 45 `f_rest` SH3 props.
- **Splat mostly disappeared after export.** Opacity pruning was applied to an MCMC
  ensemble. Keep `--min-opacity 0.0` (the default); pruning is spatial-only.
- **Scene washes out to white blur in Isaac.** Dome intensity too high for a NuRec
  splat — use the default 450, no key light.
- **fuse kept-pixel % collapses (~15–17 %).** Capture problem, not code: portrait
  orientation + fast pans (room2) vs landscape + slow walk (room3, 88 %). Re-shoot
  landscape and slow; kept-% is the health gate to check first.
- **Floor/scale wrong after refine.** A wall won the RANSAC or no ceiling was seen:
  check the per-plane gate log (`above/med/spread`), and whether the hand-height
  fallback (1.35 m) fired on a scene where the ceiling anchor should have.
- **extract refuses to run.** Stale `f_*.jpg` in the frames dir — they silently
  poison the single-camera track (mixed resolutions / numbering). Pass `--force`
  to clobber deliberately.
- **Open-spot picker chose a point outside the room.** Keep the walked-track
  restriction (`--track-radius 1.2`); the unrestricted max-clearance point was
  recon garbage beyond the walls.
- **Demo camera films only blur.** It was placed at a synthetic viewpoint inside
  splat geometry — pick a real capture pose (`video2sim.spots --camera-target ...`).
- **Stage crashes mid-pipeline.** The runner is resumable: re-run the same
  `python -m video2sim run` command; stages with existing outputs are skipped
  (`--force` to redo, `--from/--until` to slice).

## Results — v10 vs v11

Both validated on the room3 capture (62 s landscape 1080p phone walkthrough):

| | v10 | v11 (**standard**) |
|---|---|---|
| Init | sparse points, MCMC growth | **dense 350k TSDF init** (InstantSplat-style) |
| Iters | 80k (most of the gain harvested in the post-relocation settle phase) | **10k** |
| Training wall-clock | ~7× v11 (hours) | **~20 min** |
| Floaters | baseline | **38× fewer** (vs sparse init) |
| Visual | slightly better (user judgment) | approved; not worth 7× |

**v11 is the adopted standard** — this package's defaults *are* v11. Use the v10
budget (`--iters 80000` + growth) only when squeezing the last bit of sharpness
matters more than turnaround. Reference artifacts from the session:
`room3_nurec_v10.usdz` / `room3_nurec_v11.usdz` (+ matching `room3_splat_v*.ply`).

## Repository layout

```
video2sim/
  __main__.py       python -m video2sim -> cli
  cli.py            end-to-end runner (subprocess per stage, resume, venv routing)
  check_env.py      stage 0: preflight pass/fail table
  extract.py        stage 1: video -> frames (ffmpeg)
  lingbot_infer.py  stage 2: frames -> pred-*.pt (LingBot 8GB recipe)
  fuse.py           stage 3: MV-consistency filtered TSDF fusion
  refine.py         stage 4: island filter + gravity/metric alignment
  to_colmap.py      stage 5: LingBot poses + TSDF cloud -> COLMAP text model
  train.py          stage 6: gsplat MCMC trainer (+ joint pose-opt, flythrough)
  export.py         stage 7: splat.pt -> SH3 PLY -> free-space prune -> NuRec USDZ
  _nurec_convert.py    (NuRec-venv helper subprocess for the USDZ step)
  scene_prep.py     stage 8: refined mesh -> decimated collider npz
  scene.py          stage 9: splat + collider + robot -> scene.usd (Isaac venv)
  spots.py          open-spot finder + capture-camera pick
configs/default.yaml  the recipe of record (documented defaults)
tests/test_smoke.py   CPU-only smoke tests
```

## Tests

CPU-only smoke tests (se3 exp vs scipy, 62-prop PLY roundtrip, COLMAP text ↔
pycolmap, quaternion roundtrips, free-space pruning):

```bash
cd /home/perelman/AlohaMini/video2sim
/home/perelman/Basic_RL/.venv/bin/python -m pytest tests/ -q
```

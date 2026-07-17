"""video2sim — phone video to photoreal Isaac Sim scene with physics collider.

Pipeline stages (each module is importable AND runnable as ``python -m video2sim.<module>``):
  extract -> sfm -> train (gsplat) -> nurec export -> isaac scene / collider.

Interpreter contract (do NOT auto-switch — run each stage under its own venv):
  - main venv:      /home/perelman/Basic_RL/.venv/bin/python  (torch/gsplat/pycolmap/open3d/scipy)
  - NuRec export:   /home/perelman/nurec-venv/bin/python
  - Isaac scene:    /home/perelman/isaac5-venv/bin/python
  - LingBot infer:  main venv + PYTHONPATH=<lingbot fork> + ninja on PATH
"""

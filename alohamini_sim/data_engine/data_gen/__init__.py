"""Vendored AlohaMini ManiSkill data-generation package (from AlohaMini/maniskill/data_gen).

Layout mirrors the upstream repo so the original absolute imports
(``data_gen.aspire_engine.*``, ``data_gen.intern_engine.*``) keep working once the
parent directory (``alohamini_sim/data_engine``) is on ``sys.path`` —
``aspire_engine/engine.py`` inserts it automatically.

Upstream ``data_gen/__init__.py`` eagerly imported ``.tasks`` to register the ManiSkill
environments, which requires ``mani_skill``/``sapien``/``torch`` at import time. The
vendored copy keeps this ``__init__`` dependency-free so that
``alohamini_sim.data_engine.lerobot_bridge`` can import
``data_gen.aspire_engine.writer_adapter`` inside this repo's uv environment (no
simulator installed). Environment registration is instead done explicitly by the
executors (``import data_gen.tasks`` in ``aspire_engine/engine.py`` and in
``intern_engine/components/load.py``).
"""

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Import-safety tests for the vendored alohamini_sim data engine.

Two layers with different dependency budgets share ``alohamini_sim/data_engine``:

- the LeRobot bridge (``lerobot_bridge`` + ``data_gen.aspire_engine.writer_adapter``)
  must import in this repo's uv environment, WITHOUT mani_skill/sapien installed;
- the executor (``data_gen.aspire_engine.engine`` and the vendored ``agents``
  package) needs the external ManiSkill toolchain and is exercised only when
  ``mani_skill`` is importable (skipped otherwise).
"""

import importlib
import sys
from pathlib import Path

import pytest

DATA_ENGINE_DIR = Path(__file__).resolve().parents[1] / "alohamini_sim" / "data_engine"


def test_writer_adapter_imports_without_mani_skill():
    """The state-only writer must never pull in the simulator stack."""
    had_mani_skill = "mani_skill" in sys.modules
    module = importlib.import_module("alohamini_sim.data_engine.data_gen.aspire_engine.writer_adapter")
    assert len(module.STATE_NAMES) == 18
    assert len(module.ACTION_NAMES) == 16
    if not had_mani_skill:
        assert "mani_skill" not in sys.modules, "writer_adapter transitively imported mani_skill"


def test_lerobot_bridge_imports_without_mani_skill():
    """The bridge runs in the uv env: numpy + in-repo LeRobot only, no simulator."""
    pytest.importorskip("lerobot", reason="the bridge targets this repo's in-repo LeRobot API")
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    had_mani_skill = "mani_skill" in sys.modules
    module = importlib.import_module("alohamini_sim.data_engine.lerobot_bridge")
    assert callable(module.write_episodes)
    assert callable(module.convert_state_action)
    if not had_mani_skill:
        assert "mani_skill" not in sys.modules, "lerobot_bridge transitively imported mani_skill"


def test_engine_imports_with_mani_skill():
    """End-to-end import of the vendored executor in its original package layout.

    Runs only where the external toolchain is installed (e.g. the ManiSkill venv);
    in this repo's uv environment it is skipped.
    """
    pytest.importorskip("mani_skill", reason="executor needs mani-skill + sapien + torch")

    # The engine resolves its own imports by putting alohamini_sim/data_engine on
    # sys.path (upstream layout: data_gen.* / agents.* as top-level packages), but
    # importing it the first time requires the path to be present already.
    if str(DATA_ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_ENGINE_DIR))

    engine = importlib.import_module("data_gen.aspire_engine.engine")
    assert callable(engine.run_episode)
    assert callable(engine.generate_dataset)

    # The vendored agent package must be registered and resolve its URDF to the
    # vendored in-repo assets (no ~/.maniskill install required).
    agents_pkg = importlib.import_module("agents.aloha_mini")
    urdf_path = Path(agents_pkg.AlohaMiniProV2.urdf_path)
    assert urdf_path.is_file(), f"Pro agent URDF not found: {urdf_path}"

    from mani_skill.agents.registration import REGISTERED_AGENTS

    assert "aloha_mini_pro_v2" in REGISTERED_AGENTS

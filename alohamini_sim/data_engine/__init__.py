"""AlohaMini sim data engine: vendored ManiSkill episode generator + LeRobot bridge.

- ``data_gen/``: vendored copy of ``AlohaMini/maniskill/data_gen`` (aspire_engine +
  intern_engine + tasks.py) in its original package layout; executing it needs
  mani_skill/sapien/torch (see alohamini_sim/README.md).
- ``agents/``: vendored ``agents.aloha_mini`` ManiSkill agent package (SO100 +
  Pro parallel-gripper variants) with its URDF/mesh assets.
- ``lerobot_bridge``: converts engine episodes to a LeRobotDataset v3.0; runs in
  this repo's uv environment and must stay importable without the simulator.
"""

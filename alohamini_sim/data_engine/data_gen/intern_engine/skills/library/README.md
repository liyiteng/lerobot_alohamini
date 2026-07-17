# Skill Library (ASPIRE-style)

Reusable robotic knowledge distilled from validated debugging sessions, in the format of
[ASPIRE](https://research.nvidia.com/labs/gear/aspire/) (§2.2 Skill Library): each entry
stores a **failure signature** (symptoms + trace evidence), a **when-to-apply**
condition, a **repair strategy**, and a compact **code sketch** — *not* a full task
program. Entries are meant to be retrieved as in-context guidance by a coding agent (or
a human) when the matching failure signature shows up in execution traces.

Every skill here was **discovered and validated on the AlohaMini Std/Pro robots in
ManiSkill** during scripted grasp/pick-place development (2026-07). The decisive
diagnostics were per-primitive traces: IK residuals, PhysX contact-pair dumps
(`scene.get_contacts()`), commanded-vs-actual joint errors, link world-poses, and
rendered frames. This mirrors ASPIRE's robot execution engine feedback.

| skill | one-line |
|---|---|
| `base_reposition.json` | target valid but out of the arm's envelope → drive the base to a feasibility-gated station (== ASPIRE's Multi-Angle Approach) |
| `station_physics_validation.json` | a station can be kinematically fine but physically untenable → teleport + settle-check before navigating |
| `tilted_approach.json` | vertical top-down descent bottoms the palm on the object → tilt the approach axis 55–70° |
| `desc_first_ik_branch.json` | grasp-pose IK seeded from the pre-grasp lands the wrist ON the object → solve the grasp config first, seed the pre-grasp from it |
| `cartesian_line_descent.json` | joint-space interpolation between IK waypoints sweeps an arc that rakes the object → IK waypoints along the approach line |
| `linear_push.json` | push a tabletop object to target XY without grasping -> feasibility-gated station, closed horizontal gripper, 1 cm Cartesian waypoints, contact/trajectory-tuned endpoint |
| `spurious_self_collision_bits.json` | convex-hull contacts between rigidly-related links shove the arm off command → share a collision-disable bit across the subtree |
| `empty_bbox_placement.json` | scene-placement Z computed from an empty robot bbox → robust bbox (extent union, instance proxies, ancestor walk) |
| `floor_penetration_base_pinning.json` | wheel collision geometry penetrates the floor and pins NAV with huge normal impulses → spawn the root at measured wheel-rest height |
| `station_relative_approach_dir.json` | tilted approach computed from a hardcoded base origin works only at home station → derive the approach from live arm-base pose |
| `elevated_release_place.json` | tilted-axis place drags the object during surface-contact descent → release 1.5–2.5 cm above the surface and let it drop |
| `frozen_arm_base_lift_place.json` | any arm reconfiguration while holding slings the object out → freeze the arm; fine-align with the base, descend with the lift |

Loader: `library.py` (`load_skills()` returns all entries; `match(symptoms)` does a
naive keyword match over failure signatures — good enough for in-context retrieval).

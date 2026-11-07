---
name: grasp
description: RoboCasa grasp selection on observed RGB-D. Grows through verified promotions.
---

# Grasp — observed RGB-D

Purpose: `generate_grasps` / `select_grasp` on a SAM mask, with clouds in `robot0_link0`.

## Frame rules

- GraspNet returns poses in `robot0_link0`. Keep them there for `solve_ik` / `move_to_pose` /
  `plan_motion`. The pose is the public EEF (`gripper0_right_eef`), not `panda_hand` — pass
  `grasp.pose` through with no extra 10 cm / 90° map.
- Confirm the SAM overlay in `media/overlays/` before trusting the mask you passed in.
- Do not mix a grasp pose with a **world** footprint query without a transform.

## Verified entries

No verified RoboCasa entries yet. Do not copy tabletop `iterative-debugging/skills/grasp.md`.

## Ready for promotions

Add each verified pattern as a subsection with **Trigger.**, **Code.**, and **Evidence.**

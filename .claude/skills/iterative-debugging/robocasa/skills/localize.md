---
name: localize
description: RoboCasa SAM3 prompting and 3D localization. Grows through verified promotions.
---

# Localize — SAM3 on the kitchen

Purpose: choose text prompts and cameras, then back-project a mask into 3D for PandaOmron.

## Frame rules

- `localize_object` / `mask_to_point_cloud` default to the string `"robot_base"`, which matches
  **no** adapter here. Pass `get_robot_state().base_frame` (`robot0_link0`) explicitly.
- The dummy `robot0_base` is parked at world `[10, 10, 0]`. Do not use it as a frame.
- `robocasa.base_pose_is_free` and `robocasa.get_base_pose` are **world**. Transform a
  `robot0_link0` centroid before testing a footprint, or request the cloud in `world`.

## Verified entries

No verified RoboCasa entries yet. Do not copy tabletop `iterative-debugging/skills/localize.md`.

## Ready for promotions

Add each verified pattern as a subsection with **Trigger.**, **Code.**, and **Evidence.**

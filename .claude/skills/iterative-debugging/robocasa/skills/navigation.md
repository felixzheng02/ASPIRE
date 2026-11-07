---
name: navigation
description: RoboCasa wheeled-base motion and footprint queries. Grows through verified promotions.
---

# Navigation — Omron base

Purpose: move the wheeled base and ask whether a world XY is free of non-robot geometry.

## Frame rules

- `robocasa.navigate_to_pose(dx, dy, dyaw)` is **relative** P-control in the current base
  frame. It is not map-aware and it does **not** consult `base_pose_is_free`. Spawn often
  faces a counter: a forward `dx` frequently returns `ok=False`. Lateral and yaw converge.
- `robocasa.get_base_pose` and `robocasa.base_pose_is_free(x, y)` are **world**.
- `robocasa.plan_standoff_pose` / `robocasa.plan_approach_pose` inherit the cloud frame.
  Request those clouds in `world`, or transform `robot0_link0` points, before a footprint
  test.
- Arm / SAM poses stay in `robot0_link0`. Mixing frames looks like a free pose inside the
  counter, or the reverse.

## Verified entries

No verified RoboCasa entries yet. Do not copy tabletop transport snippets; they assume
`robot_base` and have no wheeled base.

## Ready for promotions

Add each verified pattern as a subsection with **Trigger.**, **Code.**, and **Evidence.**

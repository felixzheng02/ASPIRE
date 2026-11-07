---
name: manipulation
description: RoboCasa contact with articulated kitchen parts. Grows through verified promotions.
---

# Manipulation — kitchen contact

Purpose: public-API contact with articulated parts (handles, doors, drawers, switches) on
PandaOmron. Not pick-and-place transport.

## Frame rules

- Arm targets (`solve_ik`, `move_to_pose`, `plan_motion`, `move_to_joints`) are in
  `robot0_link0`. Do not hardcode world poses.
- The jaw is binary (open or close). Do not command an intermediate width.
- `execute_trajectory` densifies at 0.05 rad and closed-loop servos the last waypoint.

## Verified entries

No verified RoboCasa entries yet. Do not copy tabletop `iterative-debugging/skills/manipulation.md`
and do not invent a drawer-opening recipe here.

## Ready for promotions

Add each verified pattern as a subsection with **Trigger.**, **Code.**, and **Evidence.**

# CaP API Reference

This document is the inventory of interfaces in the generated-program runtime. It also names the
host-side adapter and provider boundaries so that implementation methods are not mistaken for
policy tools. Benchmark setup and validation remain in the individual benchmark guides.

## Surface boundaries

| Surface | Consumer | Exposure |
| --- | --- | --- |
| Shared tools | Generated `program.py` | Injected as top-level functions for every benchmark. |
| Benchmark extensions | Generated `program.py` | Injected under `libero` or `robosuite`. |
| Contract constructors | Generated `program.py` | Six allowlisted constructors are injected by name. |
| Adapter lifecycle | Harness host | Reset, native conversion, evaluation, and cleanup; never injected. |
| Provider protocols | Harness host | Perception, grasp, IK, and planning backends; never injected. |

Generated programs cannot import modules and do not receive NumPy, filesystem, network, native
simulator, reward, or success-predicate access. Arrays returned by contracts are read-only. Pass
ordinary lists or tuples when constructing array-valued inputs.

## Shared generated-program tools

`shared_atomic` calls expose one primitive capability. `shared_high_level` calls are local
compositions of those primitives. The **Advances** column describes whether a successful call can
advance environment control time.

| Layer | Tool signature | Returns | Advances | Behavior |
| --- | --- | --- | --- | --- |
| `shared_atomic` | `get_task_context()` | `TaskContext` | No | Read the authoritative task identity and language. |
| `shared_atomic` | `get_observation(camera_names=None)` | `Observation` | No | Read calibrated public cameras and robot state, optionally filtered by camera name. |
| `shared_atomic` | `get_robot_state()` | `RobotState` | No | Read joint, gripper, end-effector, frame, and timestamp state. |
| `shared_atomic` | `segment_text(camera_name, text)` | `SegmentationSet` | No | Segment objects in one current camera image using a text prompt. **`ok=True` is not a presence test** -- see [Segmentation scores](#segmentation-scores-are-not-a-presence-test). |
| `shared_atomic` | `segment_points(camera_name, points_px)` | `SegmentationSet` | No | Segment from an `N x 2` list of image-space point prompts. |
| `shared_atomic` | `mask_to_point_cloud(mask, camera_name, target_frame="robot_base")` | `PointCloud` | No | Project a public depth mask into a named frame. |
| `shared_atomic` | `crop_point_cloud(point_cloud, lower_xyz, upper_xyz)` | `PointCloud` | No | Keep points inside inclusive framed XYZ bounds, preserving aligned colors. |
| `shared_atomic` | `estimate_geometry(point_cloud)` | `ObjectGeometry` | No | Estimate a framed center, orientation, and extents from a point cloud. |
| `shared_atomic` | `generate_grasps(camera_name, mask, backend="contact-graspnet", max_candidates=5)` | `GraspSet` | No | Generate up to `max_candidates` framed candidates with the selected configured backend. On Panda embodiments the pose is the public EEF, so it can be passed to `move_to_pose` / `solve_ik` without a panda_hand offset. |
| `shared_atomic` | `solve_ik(target_pose, seed_joints=None, *, arm="primary", backend=None)` | `IKResult` | No | Solve one arm pose with the named IK backend; `None` means the benchmark's default (`pyroki`, or `curobo` on `behavior`). |
| `shared_atomic` | `plan_motion(goal, start_joints=None, *, arm="primary", strategy=None)` | `PlanResult` | No | Plan to a pose or joint goal using a `MotionStrategy`. |
| `shared_atomic` | `plan_synchronized_motion(targets, strategy=None)` | `SynchronizedPlanResult` | No | Plan same-tick motion for every arm in the current embodiment. |
| `shared_atomic` | `step(action)` | `StepResult` | One tick | Apply one normalized `RobotAction`; reward and sensitive diagnostics are removed. |
| `shared_atomic` | `execute_trajectory(trajectory, stop_on_termination=True)` | `ExecutionResult` | Many ticks | Execute a fixed-rate single- or multi-arm trajectory. Takes no tick cap: it plays every waypoint, then holds the last one for a settle tail until the joints converge -- see [Tick cost of a motion](#tick-cost-of-a-motion). |
| `shared_atomic` | `set_gripper(position, *, arm="primary")` | `ExecutionResult` | Many ticks | Drive one normalized gripper target, where `0` is closed and `1` is open. On LIBERO, Robosuite and RoboCasa only `0.0` and `1.0` are accepted -- see below. |
| `shared_atomic` | `set_grippers(positions)` | `ExecutionResult` | One tick | Command every arm's normalized gripper target in the same control tick. |
| `shared_high_level` | `localize_object(query, *, camera_name=None, target_frame=None)` | `LocalizationResult` | No | Compose text segmentation, RGB-D projection, and geometry estimation; `camera_name` defaults to the benchmark's wide camera (`agentview`, or `head` on BEHAVIOR-1K) and `target_frame` to the robot base frame. |
| `shared_high_level` | `select_grasp(grasps, strategy="top_down")` | `GraspCandidate | None | GraspSet` | No | Select `top_down` or `highest_score` from a successful grasp set, or `all_reachable` for every IK-solvable candidate as a ranked `GraspSet` -- see [Reachable grasp selection](#reachable-grasp-selection). |
| `shared_high_level` | `move_to_joints(target, tolerance=0.01, max_steps=120, *, arm="primary")` | `ExecutionResult` | Many ticks | Servo repeatedly to an absolute joint target until the measured state converges. `max_steps` is a **hard cap on executed ticks**: the call stops at exactly `max_steps` ticks, converged or not. |
| `shared_high_level` | `move_to_pose(target_pose, tolerance=0.01, max_steps=120, *, arm="primary", strategy=None)` | `ExecutionResult` | Many ticks | Plan, execute, and verify one framed end-effector pose. `max_steps` limits the **planned** waypoint count only: a plan longer than it fails with `planned trajectory exceeds max_steps` and `steps_executed` 0, without moving the arm. It does **not** cap executed ticks -- execution adds a settle tail of up to 120 ticks, so `steps_executed` can exceed `max_steps`, by the whole tail when the arm is blocked; see [Tick cost of a motion](#tick-cost-of-a-motion). On BEHAVIOR-1K a normal cuRobo reach exceeds the default of 120. |
| `shared_high_level` | `move_synchronized(targets, tolerance=0.01, max_steps=120, strategy=None)` | `ExecutionResult` | Many ticks | Plan, execute, and verify synchronized targets for all arms. `max_steps` limits the planned waypoint count only, exactly as for `move_to_pose`; executed ticks can exceed it. |
| `shared_high_level` | `open_gripper(*, arm="primary")` | `ExecutionResult` | Many ticks | Set the normalized gripper target to `1`. |
| `shared_high_level` | `close_gripper(*, arm="primary")` | `ExecutionResult` | Many ticks | Close with adapter contact-aware completion when available. |
| `shared_high_level` | `go_home(*, arm="primary")` | `ExecutionResult` | Many ticks | Servo to the arm home captured or configured by the harness. |

### Segmentation scores are not a presence test

`segment_text` returns `ok=True` and up to five ranked masks for **any** noun, including
objects that are not in the frame. The SAM3 service runs the model at a `0.0` detection
threshold rather than SAM3's own `0.5`, so nothing is ever filtered out and `results` is
never empty. A program that reads `ok` as "the object is there" will back-project a
confident-looking mask of the wrong object.

`Segmentation.score` is the number to read. It is SAM3's
`sigmoid(pred_logits) * sigmoid(presence_logit_dec)` -- a per-mask match confidence times
the model's own answer to "is this concept in this image at all". It is **not** a
mask-quality or predicted-IoU score, so a crisp, well-formed mask of the wrong object
scores *low*, not high.

Measured on RoboCasa `PickPlaceCounterToStove`, seeds 101/102/103, `agentview` and
`robot0_eye_in_hand`, 73 labelled prompts:

| cut on `score` | present kept | absent rejected | accuracy |
| --- | --- | --- | --- |
| `0.0` (the service default) | 29/29 | 0/44 | 39.7% |
| `0.5` (SAM3's own default) | 27/29 | 42/44 | 94.5% |
| `0.55` (sweep argmax) | 27/29 | 43/44 | 95.9% |

So a threshold is worth a great deal, and is still not sound. Both residual error classes
are semantic, not numeric:

- **A close noun lands on the wrong object at a high score.** On seed 101 `agentview`
  there is no sink; `"sink"` scores **0.723** and returns the plate, at the same box to
  within a pixel as `"plate"` (0.828) and `"pan"` (0.035). The present `"counter"` scores
  0.750 and `"stove"` 0.738, so the 0.027 of headroom above the false sink is not a margin
  anything should be built on.
- **A present object scores low under the wrong word.** On seed 102 a zucchini *is* on the
  counter: `"squash"` scores 0.169 and `"zucchini"` 0.127, while `"green vegetable"` on the
  same pixels scores 0.872 -- and the absent `"fish"` scores 0.549 in that same frame.

Two things a program can do:

1. **Read the score**, and compare it against `SUGGESTED_PRESENCE_THRESHOLD` (0.5) instead
   of trusting `ok`. `SegmentationSet.diagnostics` carries `top_score`,
   `best_candidate_score` (the model's best candidate *before* any filtering) and
   `service_threshold`, so the trace records the call either way.
2. **Cross-check the box.** Two different nouns that return the same box returned the same
   object, and at most one of them is right. This catches the `sink`/`plate` case, which no
   threshold does.

**On RoboCasa this is enforced by default**: `run.py` builds `Sam3Provider` with
`min_score=0.5`, so `segment_text` and `localize_object` return `ok=False` with
`NO_SEGMENTATION` when SAM3's best candidate falls below the cut, and the error names the
rejected score. Programs must handle that outcome; the remedy for a low score is another
word for the same object, not a lower threshold.

The default is per benchmark and client-side (`SEGMENTATION_MIN_SCORE` in `run.py`), not a
service setting. `:8114` is shared with LIBERO, BEHAVIOR, Robosuite and CaP-X, whose prompts
are long descriptive phrases whose score distribution nobody has measured, so they stay at
0.0. `CAP_HARNESS_SAM3_MIN_SCORE` overrides every benchmark, which is how to reproduce the
old permissive behaviour. A caller constructing `Sam3Provider()` directly still defaults to
0.0; the service can be moved for every client with `--confidence-threshold`, and one call
with a `confidence_threshold` field on `POST /segment`.

Known fallout, measured and accepted: the tool-check catalog's wrist-camera prompts now
report absence (`"cabinet"` 0.049, `"counter"` 0.398 on `robot0_eye_in_hand`, where neither
is really visible), and `connectivity_check.py` localizes its receptacle across both cameras
because the pan is not always in `agentview`.

### Gripper targets

On LIBERO, Robosuite and RoboCasa the gripper action is a **velocity**, not a position. Robosuite's
`simple_grip` controller assigns `goal_qvel`, so the sign of the command picks a direction and the
magnitude only picks how fast the jaw travels to its stop. The only widths the actuator can hold are
fully closed and fully open.

`set_gripper()` therefore accepts `0.0` and `1.0` on those embodiments and returns `unsupported`
for anything between, rather than spending its step budget waiting for a width that cannot occur.
Measured on `libero_10_swap/task_7`: `set_gripper(0.669)` and `set_gripper(0.933)` each burned
60 ticks and finished *fully open* -- 12% of the episode, spent moving away from the request.

Prefer `open_gripper()` and `close_gripper()`. A jaw cannot be pre-narrowed to reduce swing on
approach; control clearance with approach geometry instead. YAM commands a real position target and
accepts intermediate values normally.

### Reachable grasp selection

`select_grasp(grasps, "all_reachable")` returns a `GraspSet` rather than one candidate: every
candidate whose 10 cm standoff **and** grasp pose the default IK backend can solve, best
score first. A candidate whose provider reports its own `approach_distance_m` is probed at
that distance instead. `top_down` and `highest_score` are unchanged -- pure ranking, no IK,
one candidate or `None`.

It is opt-in because it is neither free nor permanent.

- **Not free.** It asks the IK backend about each candidate, so a 20-candidate set costs 20-40
  solver round trips. No simulator ticks are spent, and a candidate stops at its first
  blocking pose.
- **Not permanent.** IK is seeded from the arm's *current* configuration, so an attempt that
  moves the arm invalidates the answer. Call it again instead of reusing an old result.

It is also necessary, not sufficient. The probe is scene-free IK: cuRobo's
scene-aware planner has been measured refusing 0 of 6 standoff configurations
that its own scene-free IK accepts 6 of 6, so a candidate passing this filter
means the arm has a solution for that pose, not that the motion will execute.
Plan it with `plan_motion` and a collision-aware strategy to find out.

An empty answer is typed, never silent:

- `ok=False` with `ik_unreachable` -- candidates existed and none solved, with
  `standoff_blocked` / `grasp_blocked` counts in `details`. This is the signal to reposition
  the base, not to retry from here.
- `ok=False` with any other code -- for example `unsupported` when no IK backend is
  configured. The probe could not run, so it says nothing about reachability.

Measured on RoboCasa `PickPlaceCounterToStove`, seeds 101-150: 51% of attempted candidates
were IK-unreachable when selecting by score alone, and 4% when filtering by IK first. 139 of
885 candidates had a solvable grasp pose and an unsolvable standoff, which is why both are
checked.

### Homing

`go_home()` re-issues its joint move until the arm arrives, up to `GO_HOME_MAX_ATTEMPTS`. The
underlying `move_to_joints()` caps at `max_steps=120` and the joint servo droops on steps past
roughly 0.24 rad, so one call out of a folded pose does not converge -- measured on RoboCasa
`drawer_utensil_sort:0`, homing from a pulled-open drawer took three consecutive calls. Calls
accumulate, which is why repeating works.

`move_to_joints()` itself does **not** repeat: `max_steps` is a cap, not a guarantee, and a large
step can return `ok=False` having moved most of the way. Check the result and re-issue, or raise
`max_steps`.

### Tick cost of a motion

The same `max_steps` name bounds different things:

| Call | `max_steps` bounds | Executed ticks |
|---|---|---|
| `move_to_joints` | executed ticks, strictly | at most `max_steps` |
| `move_to_pose`, `move_synchronized` | planned waypoints, checked before any motion | waypoints + settle tail |
| `execute_trajectory` | (no parameter) | waypoints + settle tail |
| `robocasa.navigate_to_pose` | (no parameter) | until the base converges; blocked, 1200 or the rest of the episode, whichever is less |

Trajectory execution plays every waypoint once, then holds the last one until the joint error
is inside tolerance, for up to a fixed **settle tail**: `TRAJECTORY_GOAL_MAX_STEPS` = 120 ticks
on RoboCasa (`robocasa/adapter.py`, joint L2 error 0.02 rad) and
`TRAJECTORY_MAX_STEPS_PER_WAYPOINT - 1` = 119 on LIBERO-Pro, Robosuite and BEHAVIOR-1K. The tail
is separate from `max_steps` and nothing passes `max_steps` to it. RoboCasa also inserts ticks so
no joint moves more than 0.05 rad per tick, starting from the trajectory's start state; planned
paths are already that dense, so this adds ticks only to sparse hand-built trajectories.

In free space the tail is short -- measured 2-29 ticks on RoboCasa -- so the difference is easy
to miss. When the arm is blocked it never converges and pays the whole tail: measured on RoboCasa,
`move_to_pose(max_steps=150)` against the stove panel ran 32 waypoints + 120 settle = **152**
ticks and returned `ok=False`. So the worst case of one blocked call is `waypoints + 120`, and
retrying a blocked move pays it again. Budget from
`get_task_context().metadata["episode_steps_remaining"]`, not from a sum of `max_steps`; when one
move needs a hard tick cap, use `move_to_joints`.

RoboCasa base motion is charged to the same budget. Every tick `robocasa.navigate_to_pose` runs
is one episode step: it counts in `episode_steps_used`, is scored, and appears in
`episode/steps.jsonl` and the videos. A base tick is recorded with the arm command it carried,
which is "hold the measured joints"; the base velocity has no field in the step record. A move
that converges is cheap -- measured 9 ticks for 0.08 m sideways and 7 for a 0.2 rad turn. A
blocked move is not. The base loop gives up only after 1200 ticks and the call takes no tick
cap. RoboCasa's episode is 1000 ticks, so at the default horizon **one blocked base move runs
until the episode is truncated**. It returns `ok=False` with `truncated=True` and the real
`steps_executed`, and `episode_steps_remaining` then reads 0. (With more than 1200 ticks left it
returns `ok=False` after 1200, not truncated.) `robocasa.base_pose_is_free` can rule out an occupied
goal before driving; it does not check the path.

Runs recorded before 2026-11-17 did not record base ticks at all: `episode_steps_used` stayed
unchanged across a base move while the episode horizon still counted it, and those ticks are
missing from their `steps.jsonl` and videos. Do not compare step counts or artifacts of
base-driving runs across that date.

### Motion strategies

`MotionStrategy()` defaults to PyRoki IK plus joint interpolation on LIBERO-Pro and Robosuite; on
BEHAVIOR-1K the harness sets the default to cuRobo for IK, trajectories and integrated pose
planning, and `solve_ik(backend=None)` follows that default (PyRoki is not registered there). Set
`trajectory_planner="curobo"` for cuRoboV2 c-space planning, or
`pose_planner="curobo-integrated"` for integrated pose planning. Requested unavailable backends
return typed failures; the harness never falls back implicitly. `move_to_joints()` and `go_home()`
are direct servo operations rather than collision-aware plans. Synchronized pose planning requires
targets for every arm and defaults to cuRobo-integrated.

### Cartesian pose inputs

`move_to_pose` accepts a Python `Pose` or a mapping with `position` in metres,
`frame`, and exactly one orientation field:

- `quaternion_wxyz`: a unit quaternion `[w, x, y, z]`.
- `rpy_deg`: `[roll, pitch, yaw]` in **degrees**, using fixed-axis XYZ rotations
  in the named pose frame: `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`.

The MCP `cap_move_to_pose` tool accepts the same mapping as `target_pose`:

```json
{
  "arm": "left",
  "target_pose": {
    "position": [0.5, 0.2, 0.9],
    "rpy_deg": [180, 0, 90],
    "frame": "world"
  },
  "strategy": {"ik_solver": "mink", "trajectory_planner": "curobo"}
}
```

The numbers above illustrate the format; select targets from the observed scene.
MCP pose targets in `cap_move_synchronized` accept either orientation format too.
Both formats describe absolute orientations, not relative rotations. Missing,
ambiguous, malformed, or nonfinite pose inputs are rejected before planning.
RPY is converted to the existing quaternion `Pose` internally. Returned poses
continue to use `quaternion_wxyz`.

## Benchmark extensions and differences

Each run injects only the namespace matching its selected benchmark. Metadata mappings are
read-only and intentionally retain benchmark-specific fields rather than claiming one common
schema.

| Layer | Tool | Current fields |
| --- | --- | --- |
| `extension` | `libero.get_task_metadata()` | `family`, `init_state_count`, `init_state_index`, `language`, `seed`, `suite`, `task_id`, `task_name`, `task_ref` |
| `extension` | `libero.get_controller_metadata()` | Action dimension/bounds, arms, cameras, frequency/period, controller, gripper and joint-command semantics, action modes |
| `extension` | `robosuite.get_task_metadata()` | `arms`, `camera_names`, `environment`, `family`, `language`, `robots`, `suite_name`, task identity |
| `extension` | `robosuite.get_controller_metadata()` | Action dimension/bounds, arms, base frame, cameras, frequency/period, controller, controllable grippers, action modes |
| `extension` | `robocasa.get_task_metadata()` | `arms`, `camera_names`, `environment`, `family`, `language`, `robots`, `suite_name`, task identity |
| `extension` | `robocasa.get_controller_metadata()` | Action dimension, arms, base frame `robot0_link0`, cameras, frequency/period, controller, gripper semantics, action modes, `mobile_base`, `joint_limits` |
| `extension` | `robocasa.get_base_pose()` | `Pose` of the mobile base in `world`. Proprioception a real mobile robot also has, not simulator state |
| `extension` | `robocasa.navigate_to_pose(dx, dy, dyaw)` | Closed-loop RELATIVE base motion in the current base frame. No collision checking by design: a blocked path fails to converge and returns `ExecutionResult(ok=False)` with a `base_translation_m` diagnostic. Forward motion from the spawn pose is usually blocked by a counter; lateral and yaw converge. Every tick is an episode step and the call takes no tick cap, so at the default 1000-tick horizon a blocked move runs until the episode is truncated (`truncated=True`) -- see [Tick cost of a motion](#tick-cost-of-a-motion) |
| `extension` | `robocasa.base_pose_is_free(x, y, yaw=None)` | Whether a PandaOmron footprint at world `(x, y)` is empty of scene geometry. `yaw` defaults to the base's CURRENT yaw, not zero: the footprint is a rectangle offset 0.20 m behind `robot0_link0` and this robot spawns near 90°, so a zero default reports the robot's own pose blocked. Same frame as `robocasa.get_base_pose`. Returns a boolean, not a map. Arm poses and default SAM clouds are `robot0_link0` — transform before mixing |
| `extension` | `robocasa.plan_standoff_pose(support_cloud, object_cloud, standoff_m=0.3)` | Pure geometry: `(x, y, yaw)` outside the support hull's nearest edge, facing the object. Both clouds must share a frame, which is the frame of the answer |
| `extension` | `robocasa.plan_approach_pose(object_cloud, distance_m=0.7)` | Pure geometry: `(x, y, yaw)` `distance_m` short of the object along the current line of sight. Answer is in the cloud's frame |
| `extension` | `behavior.get_task_metadata()` | `activity_name`, `arms`, `camera_names`, `family`, `instance_id`, `language`, `scene_model`, `suite_name`, `support_prompt`, `target_prompt`, task identity |
| `extension` | `behavior.get_controller_metadata()` | Action dimension/bounds, arms, base frame `odom`, `base_pose`, cameras, frequency/period, controller, controllable grippers, `torso_positions`, `torso_lower_bounds`, `torso_upper_bounds` |
| `extension` | `behavior.get_base_pose()` | `(x, y, yaw)` of the base footprint in `odom` |
| `extension` | `behavior.navigate_to_pose(x, y, yaw, *, planner="curobo", tolerance_m=0.05, tolerance_rad=0.1, max_steps=1500)` | Drives the holonomic base to an `odom` pose; `planner` is `curobo` (collision-aware plan) or `servo` (direct, for in-place turns and short hops); returns `ExecutionResult` with `moved_x`, `moved_y`, `moved_yaw` diagnostics plus, for `curobo`, `route` (`direct`, `traversability` or `none`), `hops`, `servo_hops` and `direct_status` |
| `extension` | `behavior.move_torso(target, *, tolerance=0.01, max_steps=300)` | Servo the four torso joints to `target` (radians); returns `ExecutionResult` |
| `extension` | `behavior.reset_torso()` | Return the torso to the instance's reset pose; returns `ExecutionResult` |
| `extension` | `behavior.plan_standoff_pose(support_cloud, object_cloud, standoff_m=0.3)` | Pure geometry: `(x, y, yaw)` in `odom` on the support surface's nearest edge, facing the object |
| `extension` | `behavior.plan_approach_pose(object_cloud, distance_m=0.7)` | Pure geometry: `(x, y, yaw)` in `odom` `distance_m` short of a free-standing object along the current line of sight, facing it |
| `extension` | `behavior.base_pose_is_free(x, y)` | Whether the base footprint fits at that `odom` position according to the scene's traversability map (`True` when the scene has no map); `navigate_to_pose` refuses a blocked goal before planning, so test candidate approach poses with this first |

| Benchmark | Arms | Public base frame | Cameras | Control notes |
| --- | --- | --- | --- | --- |
| LIBERO-Pro | `primary` | `robot_base` | `agentview`, `robot0_eye_in_hand` | Seven-joint Panda plus normalized gripper at 20 Hz. |
| Robosuite | `primary`, optionally `secondary` | `robot0_base` | Agent view and one eye-in-hand camera per robot | Single- or bimanual Panda joint control at 20 Hz. |
| BEHAVIOR-1K | `primary` (left), `secondary` (right) | `odom`, fixed where the base stood at instance load | `head`, `left_wrist`, `right_wrist` | R1 Pro seven-joint arms plus normalized grippers at 30 Hz; the holonomic base and the torso move only through the `behavior.*` extensions, and every pose stays valid after the base drives. |

`get_task_context().language` is the authoritative per-episode instruction.

## Injected constructors and returned contracts

Programs may `import math` (the only permitted import, without aliasing); every other name is
injected by the harness.

| Constructor | Signature |
| --- | --- |
| `ArmCommand` | `ArmCommand(mode, target, gripper_position=None)` |
| `MotionStrategy` | `MotionStrategy(ik_solver="pyroki", trajectory_planner="interpolation", pose_planner="composed")` |
| `Pose` | `Pose(position, quaternion_wxyz, frame)`; quaternion order is `wxyz` |
| `RobotAction` | `RobotAction(arms)` |
| `Trajectory` | `Trajectory(joint_positions, dt_s, joint_names, planner, collision_aware, expected_start, arm="primary", gripper_positions=None, embodiment=...)` |
| `SynchronizedTrajectory` | `SynchronizedTrajectory(joint_positions, dt_s, joint_names, planner, collision_aware, expected_start, gripper_positions=None, embodiment=...)` |

`embodiment` defaults to `"robosuite"`, which every other embodiment's adapter rejects at
execution. A hand-built `Trajectory` or `SynchronizedTrajectory` must pass
`embodiment=get_robot_state().embodiment`; prefer `move_to_pose()` / `move_to_joints()`, which
build it for you.

Returned objects expose validated attributes:

| Contract | Important public data |
| --- | --- |
| `TaskContext` | Suite, task id/name, language, family, and public metadata. |
| `Observation` | Camera mapping, `RobotState`, task context, and timestamp. |
| `RobotState` | Per-arm 7-DoF joints, velocities, end-effector poses, grippers, names, and base frame. |
| `SegmentationSet` | `ok`, masks, labels, scores, boxes, diagnostics, and typed error. |
| `PointCloud` / `ObjectGeometry` | Framed points; estimated pose, extents, and point count. |
| `GraspSet` / `GraspCandidate` | Framed candidate poses, scores, widths, diagnostics, and typed error. |
| `IKResult` | `ok`, arm, joint solution, diagnostics, and typed error. |
| `PlanResult` / `SynchronizedPlanResult` | `ok`, trajectory, diagnostics, and typed error. |
| `LocalizationResult` | `ok`, segmentation, point cloud, geometry, diagnostics, and typed error. |
| `StepResult` | Public observation, terminal flags, diagnostics, and typed error; reward is stripped. |
| `ExecutionResult` | `ok`, stable status, executed steps, per-arm final errors, final observation, terminal flags, diagnostics, and typed error. |

## Host-only adapter and provider interfaces

These names describe Python integration boundaries and are not generated-program tools.

- `EnvironmentAdapter` defines embodiment identity, reset, task/observation/state reads, one-tick
  stepping, trajectory execution, single- and same-tick multi-gripper control, and cleanup.
- The simulation runner additionally uses `check_success()` after program execution. Safe
  namespaced extensions use `get_task_metadata()` and `get_controller_metadata()`.
- Feature-detected adapter hooks include `get_planning_context()`, `action_bounds`, `ik_request()`,
  and contact-aware `close_gripper()`.
- `native_env`, `native_step()`, `compute_reward()`, and native timing/evaluation state remain
  validation-only and are never injected into a program.
- Provider protocols cover segmentation, grasp generation, IK, joint-trajectory planning, and
  integrated pose/synchronized planning.

The current source layout is deliberately recorded rather than hidden: LIBERO metadata is defined
in `libero/extensions.py` and forwarded by its adapter; Robosuite metadata lives on
`RobosuiteAdapter`. Concrete adapters remain host-side even
where a package currently re-exports their Python classes.

## Current limitations

- For cross-benchmark projection, pass `target_frame=get_robot_state().base_frame` because
  `mask_to_point_cloud()` currently defaults to LIBERO's `robot_base`.
- `segment_points()` does not expose provider point labels.
- `execute_trajectory(..., stop_on_termination=False)` returns an unsupported failure.
- Public step and execution results remove reward, success, and nested sensitive diagnostics.

## Real YAM

See [Real YAM](yam-real.md) for station setup, physical motion authorization,
arm aliases, local Mink IK, and recording. The shared API remains the same.

## System 2

The [MCP server](system2.md) exposes selected CAP tools, live observations,
job status, and operator steering.

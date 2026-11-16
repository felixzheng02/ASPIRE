# Grasp-only benchmark — results

`PickPlaceCounterToStove`, seeds 101–150 (TRAIN scene group), top 3 candidates per episode,
17.8 minutes total. Raw data in `results.jsonl`, machine summary in `summary.json`.

## Headline

**2 of 50 episodes picked the object up (4%).** On a stricter reading, **1 of 50 (2%)** — see
"the second success is probably spurious" below.

## Where the 48 failures happen

| | episodes |
|---|---:|
| reached the pose, closed, object did not come up | 24 |
| every one of the top 3 candidates was IK-unreachable | 21 |
| no mask, no candidates, or a simulator error | 3 |
| picked | 2 |

Across all 138 individual attempts, **70 (51%) were IK-unreachable** and never moved the arm.
`generate_grasps` and `select_grasp` know nothing about the robot, so roughly half of what the
grasp API proposes cannot be reached from where the arm is standing, and in 21 episodes all
three top-scoring candidates were infeasible.

## This is not a clean measurement of GraspNet

Two other components are inside the loop and both degrade the number:

**Segmentation.** Median mask score across the 50 episodes is **0.231**; 30 of 49 are below
0.5 and 21 are below 0.1. The object noun parsed from the instruction frequently does not
retrieve the object — RoboCasa names things like `marshmallow`, `squash`, `corn`, and SAM3
scores those poorly even when the item is in frame. Restricting to the 19 episodes with a
confident mask (≥ 0.5) gives **1 pick out of 19**, so better segmentation alone does not
rescue the number — but 6 of those 19 still died on reachability.

**Control.** Only 5 of 68 executed attempts had `move_to_pose` report `ok` on the grasp pose.
That figure overstates the problem: the adapter's settle loop converges in joint space while
`CapApi` judges Cartesian error, so moves that physically arrive can still report
`EXECUTION_FAILED`. This is why "picked" is decided by the object's measured rise and not by
the move's own flag.

The gripper meets nothing. `_measured_gripper` is normalised with 1.0 open and 0.0 closed
(`adapter.py`, `qpos[0] / 0.04`), so `jaw_after_close ≈ 0.04` means the fingers travelled to
about 1.6 mm of fully shut. Across the 87 executed attempts of the `--ik-filter` run, 72
closed flat (jaw < 0.10) and the median object rise is −0.0002 m: the object was never
touched.

## The second success is probably spurious

| seed | object | mask score | rank | grasp score | measured rise | jaw after close |
|---|---|---:|---:|---:|---:|---:|
| 136 | garlic | 0.606 | 1 | 0.240 | 0.069 m | 0.040 |
| 144 | marshmallow | **0.006** | 0 | 0.278 | **0.411 m** | 0.000 |

Seed 144 reports a 0.411 m rise after a 0.12 m lift, which is physically impossible. Its mask
score of 0.006 means SAM3 found nothing resembling a marshmallow, so the "before" and "after"
masks are almost certainly different parts of the scene and the centroid difference is
meaningless. Seed 136 looks genuine: a confident mask, a plausible 6.9 cm rise, and a jaw that
stalled on an object.

That is a limitation of this benchmark's own success test, not of the grasp API: verifying a
pick by re-segmenting the object is only as trustworthy as the segmentation, and here the
segmentation is weak. A tighter criterion — requiring the rise to fall in a plausible
0.05–0.20 m band — gives 1 of 50.

## Caveats

One task, one scene group, one seed range, three candidates per episode, no retry on a
different mask, no base repositioning. Measured after the four fixes committed in this
session (`85ef519`, `c2d31f3`, `195627c`, `0bf8762`, `b219f9f`); in particular grasp poses are
now deterministic and in the public EEF frame, so this number is reproducible in a way it
would not have been before.

---

# Second run: `--ik-filter`

Same 50 seeds, same everything, except candidates are ranked best-first and only those
`solve_ik` can reach are attempted. Raw data in `ik_filter/`.

| | baseline | `--ik-filter` |
|---|---:|---:|
| picked (as scored) | 2 / 50 | **5 / 50** |
| picked (plausible 0.05–0.20 m rise) | 1 / 50 | **3 / 50** |
| attempts made | 138 | 91 |
| of those, IK-unreachable | 70 (51%) | **4 (4%)** |
| attempts that actually moved the arm | 68 | 87 |
| wall clock | 17.8 min | 21.9 min |

Filtering works as intended: it converts 70 wasted attempts into 87 real ones and roughly
doubles the pick rate on either success criterion. It is not a fix for grasping — it stops
the attempt budget being spent on poses the arm cannot reach.

## Where the remaining 45 failures are

Reachability does not disappear, it moves. **16 of the 48 episodes that produced candidates
had no feasible candidate at all** among up to 20 — a third of episodes where the grasp API
proposes nothing the arm can reach. Restricting to the 32 episodes with at least one feasible
candidate, the pick rate is **5 / 32 (16%)**.

`no_grasps` rose from 2 to 18 because the summary counts "no candidate worth attempting",
which now includes those 16 zero-feasible episodes; `episodes_with_no_candidates` (GraspNet
genuinely returned nothing) is 2, essentially unchanged.

## The successes

| seed | object | mask | feasible/total | rank | score | rise | jaw |
|---|---|---:|---:|---:|---:|---:|---:|
| 119 | squash | 0.089 | 2/11 | 1 | 0.181 | **+0.568** | 0.038 |
| 135 | sweet potato | 0.077 | 3/20 | 2 | 0.291 | +0.076 | 0.041 |
| 136 | garlic | 0.606 | 2/11 | 1 | 0.240 | +0.069 | 0.040 |
| 137 | broccoli | 0.844 | 3/20 | 2 | 0.257 | +0.104 | 0.038 |
| 144 | marshmallow | 0.006 | 3/20 | 0 | **+0.411** | 0.000 | 0.000 |

Seeds 119 and 144 report rises of 0.568 m and 0.411 m after a 0.12 m lift, which is
impossible, and both have near-zero mask scores (0.089, 0.006) — the re-segmentation found a
different part of the scene, so those two are measurement artifacts, not picks. The three
with plausible rises (135, 136, 137) have jaw widths around 0.04, which on this normalised
scale is almost fully closed, not a jaw stalled on an object. Why those three lifted anyway
is open.

Note that **3 of the 5 "successes" came from rank 1 or 2**, not the top-scoring candidate, so
trying more than one candidate is doing real work.

## What this says about the grasp API

Of the 50 episodes, on the stricter criterion: 3 picked, 16 had no reachable candidate, 2 had
no candidate at all, and 29 reached a pose, closed, and left the object where it was. The
dominant failure is still grasp quality — but reachability is a close second and is purely an
API-shape problem: `generate_grasps` and `select_grasp` never see the robot, so a third of
scenes yield 20 candidates and nothing attemptable.


---

# Correction (2026-11-15)

Two premises above were wrong. Both were found by re-reading the data, no simulator needed.

**1. The jaw reading was inverted.** `RobocasaAdapter._measured_gripper` returns
`qpos[0] / 0.04` clipped to [0, 1], so **1.0 is open and 0.0 is closed** — confirmed live
earlier in the same session, where `open_gripper()` measured 0.986 and `close_gripper()`
0.014. The original text read `jaw ≈ 0.04` as "the jaw stayed open, so the fingers met
something". It means the opposite: the fingers closed to within ~1.6 mm of flat and met
nothing. Recomputed over the 87 executed attempts of the `--ik-filter` run: **72 closed flat
(jaw < 0.10), 15 never closed (jaw > 0.30), median object rise −0.0002 m.**

The question this changes: not "what is the gripper closing on" but "why is nothing between
the fingers".

**2. Fifteen of the 87 recorded attempts never happened.** All 15 of the jaw > 0.30 attempts
are rank 2 — the third attempt — and all have `approach_ok`, `grasp_ok` *and* `lift_ok` false:
nothing moved. The cause is episode-horizon exhaustion. `adapter._truncated()` is
`_elapsed_steps >= horizon` with `horizon=1000`, and `_elapsed_steps` resets only on
`reset()`. Three attempts of (30 open ticks + approach + 30 close ticks + lift) exhaust the
budget, after which every step returns truncated, `execute_trajectory` bails after one tick,
and the gripper is never commanded.

**This is a bug in `grasp_bench.py`, not in the API.** `attempt_grasp` ignores
`ExecutionResult.ok` and the truncation flag and records these as ordinary `not_picked`
attempts. So "29 episodes reached a pose and closed on nothing" is overstated — in up to 15
of those attempts the arm never moved at all. The benchmark should reset or budget per
attempt, and should record truncation.

Unaffected by either correction: the reachability numbers (51% unreachable, 16/48 scenes with
no feasible candidate), the pick counts, and the grasp-frame conclusion.
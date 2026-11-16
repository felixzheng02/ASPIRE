# Grasp-only benchmark

How often does the grasp API actually pick the object up?

One episode per seed, all through the public RoboCasa tool surface:

1. `get_task_context()` → parse the object noun out of the instruction.
2. `segment_text("agentview", noun)` → best mask.
3. `generate_grasps(..., max_candidates=20)` → candidates, sorted by score.
4. For the top three in order: `solve_ik` on the standoff and the grasp pose, then
   `move_to_pose` to a 10 cm standoff, `move_to_pose` to the grasp pose, `close_gripper`,
   and `move_to_pose` 12 cm up.
5. Re-segment the object and compare its cloud centroid. A rise above 5 cm counts as picked.
   Stop at the first success.

Nothing else. No placing, no base motion, no task success, no videos. The simulator's
`evaluate` channel is never consulted — it scores placing, which is not what this measures,
and it is host-only anyway. "Picked" is decided entirely from what a generated program can
see.

## Running it

Needs the simulator serving the task, plus SAM3, GraspNet and PyRoki. From `aspire/`:

```bash
PYTHONPATH="" .venv-robocasa/bin/python examples/robocasa/grasp_bench/grasp_bench.py \
    --seeds 101-150
```

Options: `--task` (default `PickPlaceCounterToStove`), `--seeds` (a range like `101-150` or a
comma list), `--attempts` (default 3), `--out` (default `out/` beside the script).

Seeds ≥ 101 draw from RoboCasa's TRAIN scene group, 1–100 from TEST; the two are disjoint.

## Output

- `out/results.jsonl` — one line per episode: the instruction, the object noun, the mask
  score, the candidate count, and every attempt with its score, whether IK was reachable,
  whether each move converged, the jaw width after closing, and the object's measured rise.
- `out/summary.json` — episode count, pick count, success rate, which rank succeeded, and a
  tally of failure reasons.

## What this does and does not measure

It measures the chain as a program would use it, which means **it is not a clean measurement
of GraspNet alone**. Three other things can sink an episode:

- **Segmentation.** The grasp is only as good as the mask it is given, and `segment_text`
  has no presence test — a low score does not stop it returning a confident-looking wrong
  mask. `mask_score` is recorded per episode so these can be separated afterwards.
- **Reachability.** `generate_grasps` and `select_grasp` know nothing about the arm, so a
  top-scoring candidate is often IK-infeasible. Those attempts are recorded as
  `unreachable` and cost no simulator time.
- **Control.** `move_to_pose` judges success in Cartesian space while the adapter's settle
  loop converges in joint space, so `approach_ok` / `grasp_ok` can read `False` on a move
  that physically arrived. That is why `picked` is decided by the object's rise rather than
  by the move's own `ok` flag.

Read `failure_reasons` and the per-attempt `outcome` together before attributing a number to
any one component.

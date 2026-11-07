# RoboCasa Stage 1b — Evolutionary Search (optional)

Follow this runbook **only when** `$RUN_ROOT/campaign.json` has `"evosearch": true`. If the
key is missing or `false`, skip this file and go to the unseen-check seed (126).

This is **not** a separate campaign and **not** the tabletop evosearch skill. Do not follow
`.claude/skills/evosearch/SKILL.md` (different worker count, different seed protocol,
`supervise_services.sh`). Stay on this kitchen, this simulator, and this worker.

## When it runs

After Stage 1 Step 3 has copied a program to `$TASK_DIR/fix_code.py`, and **before** the
single unseen-check on seed 126. Same worker, same `server.py`, same GPU. Do not start a
second simulator. Do not CLOSE the live one. Never kill SAM3, PyRoki, GraspNet, or cuRobo.
Never `rm`. Never `pkill`.

## Budget

| | |
|---|---|
| Candidates K | **4** (`candidate_A` … `candidate_D`) |
| Rounds T | **3** (`iter_00`, `iter_01`, `iter_02`) |
| Eval seeds | development **101 102 103** only |
| Parallelism | `--parallel-per-gpu 1` and `run-batch --workers 1` |

One kitchen cannot interleave two episodes. **`--parallel-per-gpu` MUST be 1.** Never raise
`--workers` above 1.

To raise K later, add `candidate_E/code.py` and further names and pass them to
`evosearch_eval.py`. Keep `--parallel-per-gpu 1`. Do not raise `--workers`.

## Procedure

1. Record the freeze-loop development pass-count (how many of 101 102 103 `task_success` on
   the selected `fix_code.py`). That is the number to beat.
2. Create `$TASK_DIR/debug/evosearch/iter_00/`. Copy `fix_code.py` **verbatim** to
   `candidate_A/code.py`. Write three more candidates (`B` `C` `D`), each a distinct
   hypothesis — not four paraphrases. Same public-API / provenance rules as Stage 1.
3. Evaluate the matrix with the existing evaluator (do not hand-roll a seed loop):

```bash
PYTHONPATH="" python3 .claude/skills/evosearch/scripts/evosearch_eval.py \
  --iter-dir "$TASK_DIR/debug/evosearch/iter_00" \
  --benchmark robocasa --suite "$SUITE" --task-id "$TASK_ID" \
  --init-mode saved --trial-seeds 101 102 103 \
  --sim-gpus 0 --parallel-per-gpu 1 \
  --cap-harness "$CAP_HARNESS"
```

4. Read `iter_summary.json`. Diagnose from *your* candidate run artifacts only. For rounds
   `iter_01` and `iter_02`, seed the next `candidate_A` from the current best and write K=4
   again. Repeat the same `evosearch_eval.py` command with the new `--iter-dir`.
5. After T=3 (or sooner if one candidate passes 3/3 and you would not change it), take the
   best development pass-count. If it **beats** the freeze-loop number, copy that
   `code.py` over `$TASK_DIR/fix_code.py`. If it ties or loses, keep the freeze-loop
   program. Do not assemble a chimera that never ran as one file.
6. Return to the worker prompt Step 4: measure seed **126** once on whatever `fix_code.py`
   now is. Never retune on 126. Never run held-out 1 2 3.

## Frame rules (unchanged)

Arm / SAM / GraspNet / cuRobo poses are `robot0_link0`. `base_pose_is_free` is **world**.
`navigate_to_pose` is relative P-control and does not consult the footprint query.
`examples/robocasa/connectivity_check.py` is an integration gate, not a policy — do not copy
it in as a candidate.

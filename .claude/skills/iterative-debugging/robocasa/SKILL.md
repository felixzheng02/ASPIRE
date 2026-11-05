---
name: iterative-debugging-robocasa
description: "Frozen-program CaP fix loop on RoboCasa (PandaOmron kitchen, out-of-process sim). Same product as the tabletop loop — write a program, debug on development kitchens, freeze it, replay on held-out kitchens — with embodiment constraints: worker-owned simulator, shared SAM3, run-batch --workers 1. First campaign is OpenDrawer on seeds 101 102 103 / 126 / 1-3."
---

# Iterative Debugging (Fix Loop) on RoboCasa

The parent skill ([../SKILL.md](../SKILL.md)) writes one whole program per task, freezes it as
`fix_code.py`, and replays that frozen program on held-out seeds. **That product applies here.**
BEHAVIOR's sibling ([../behavior/SKILL.md](../behavior/SKILL.md)) does not: it grows a per-seed
policy and freezes a skill library. Do not mix the two.

What is different is the **embodiment**, not the scientific claim:

| Constraint | Consequence |
|---|---|
| Simulator is another process | The Stage 1 worker starts `server.py` and keeps a PID file. `cap-harness` close does not kill it. |
| One kitchen at a time | `run-batch --workers 1 --recycle-after 0` only. Never raise `--workers`. |
| SAM3 is a shared process on `:8114` | Start-if-idle with `scripts/ensure_sam3.sh`. Never kill it. `GET /healthz` before episodes. |
| No cartesian stack | Do not build a strategy on tools that return unsupported. Joint motion, interpolated trajectories, gripper 0/1, and `robocasa.navigate_to_pose` are the motion surface. |
| Spawn faces a counter | Forward base motion often fails; lateral and yaw converge. Reach from spawn is ~0.85 m. |

This folder has no `skills/` library and no verifier. The worker writes `findings.md`. Promotion
into a shared library is a later campaign.

## Setup

Working directory for harness commands is the CaP harness repo (`aspire/`, the directory with
`pyproject.toml`). `WORKSPACE_ROOT` is its parent (contains `aspire/`, `robocasa/`, and
`scripts/ensure_sam3.sh`).

1. `aspire/scripts/bootstrap_robocasa.sh` once per machine (see [docs/robocasa.md](../../../../docs/robocasa.md)).
2. SAM3 weights are gated. Request access, then `aspire/.venv-sam3/bin/hf auth login` if
   `ensure_sam3.sh` says to. Do not paste tokens.
3. From `$WORKSPACE_ROOT`: `./scripts/ensure_sam3.sh` (start-if-idle; never kills). Confirm
   `GET /healthz` on `http://127.0.0.1:8114` is 200.
4. Read [docs/api-reference.md](../../../../docs/api-reference.md) (the `robocasa.*` rows),
   [docs/robocasa.md](../../../../docs/robocasa.md), [docs/run-artifacts.md](../../../../docs/run-artifacts.md),
   and [main-agent-prompt.md](main-agent-prompt.md).
5. Do not start `aspire/scripts/supervise_services.sh` for this campaign.

## Seeds

A seed selects the kitchen. `scene_group_for_seed()` cuts TRAIN at ≥101 and TEST at 1–100.
This campaign is **tiny** and stays on the right side of that cut. Widening later uses the same
cut with the full Robosuite-shaped lists (development 101–125, unseen-check 126–130, held-out
1–100) recorded in `campaign.json`.

| Partition | Seeds | Rule |
|---|---|---|
| Stage 1, development | 101 102 103 | Debug here. Every program version runs on all three. |
| Stage 1, unseen-check | 126 | Measure the selected program once. Never debug against it. |
| Stage 2, held-out | 1 2 3 | Coordinator only, frozen `fix_code.py`. TEST kitchens. |

`--init-mode saved`. All variation comes from `--seed`.

## Campaign layout

```
$RUN_ROOT/robocasa/OpenDrawer/task_0/
├── code/v00_initial.py … v<NN>_<slug>.py  CHANGELOG.md
├── debug/explore/ initial/ attempts/ holdout/ logs/ blocked/
├── fix_code.py
├── findings.md
└── simulator.pid
```

`$RUN_ROOT` comes from
`python3 .claude/skills/iterative-debugging/scripts/init_run.py` as in the parent, with this
campaign's seed lists (see [main-agent-prompt.md](main-agent-prompt.md)).

## Run order

1. Write `tasks.json` with the single case
   `{"benchmark": "robocasa", "suite": "OpenDrawer", "task_id": 0}` and initialize the campaign
   root as in [main-agent-prompt.md](main-agent-prompt.md).
2. `./scripts/ensure_sam3.sh` from `$WORKSPACE_ROOT`.
3. Dispatch one Stage 1 worker with [subagent-prompt.md](subagent-prompt.md). The worker starts
   the simulator, explores seed 101, writes and scores versions on 101 102 103, freezes
   `fix_code.py`, measures once on 126, and returns.
4. Coordinator runs Stage 2 with [scripts/run_validation.py](../scripts/run_validation.py)
   `--workers 1 --init-mode saved --seeds 1 2 3` against `fix_code.py`, reusing the worker's
   simulator. Do not dispatch a verifier. Do not promote skills.
5. Use [clean-task-slate.md](clean-task-slate.md) before a rerun.

This skill is OpenDrawer only. A later campaign may run `PickPlaceCounterToStove` after a
simulator restart with `--task` matching that suite.

---
name: iterative-debugging-robocasa
description: "Frozen-program CaP fix loop on RoboCasa (PandaOmron kitchen, out-of-process sim). Same product as the tabletop loop — write a program, debug on development kitchens, freeze it, replay on held-out kitchens — with embodiment constraints: worker-owned simulator, shared SAM3, run-batch --workers 1. Local skill library + serialized verifier; evolutionary search is --evosearch at init, default off. Suite comes from tasks.json; the instruction comes from get_task_context().language."
---

# Iterative Debugging (Fix Loop) on RoboCasa

The parent skill ([../SKILL.md](../SKILL.md)) writes one whole program per task, freezes it as
`fix_code.py`, and replays that frozen program on held-out seeds. **That product applies here.**
BEHAVIOR's sibling ([../behavior/SKILL.md](../behavior/SKILL.md)) does not: it grows a per-seed
policy and freezes a skill library. Do not mix the two.

The paper loop on this kitchen is the tabletop one: workers propose skills → a verifier
reproduces from text alone → the coordinator promotes into
[skills/](skills/). Evolutionary search is an **optional extra search over programs**, not a
different product.

What is different is the **embodiment**, not the scientific claim:

| Constraint | Consequence |
|---|---|
| Simulator is another process | The Stage 1 worker starts `server.py` and keeps a PID file. `cap-harness` close does not kill it. |
| One kitchen at a time | `run-batch --workers 1 --recycle-after 0` only. Never raise `--workers`. |
| SAM3 is a shared process on `:8114` | Start-if-idle with `scripts/ensure_sam3.sh`. Never kill it. `GET /healthz` before episodes. |
| PyRoki is a shared process on `:8116` | Start-if-idle with `scripts/ensure_pyroki.sh`. Never kill it. `GET /healthz` before episodes. `solve_ik` / `move_to_pose` are live. |
| GraspNet is a shared process on `:8115` | Start-if-idle with `scripts/ensure_graspnet.sh`. Never kill it. `generate_grasps` / `select_grasp` are live. Not required for every `cap-harness run`. |
| cuRobo is a shared process on `:8118` | Start-if-idle with `scripts/ensure_curobo.sh`. Never kill it. Collision-aware `plan_motion` is live on observed RGB-D. Not required for every `cap-harness run`. |
| Spawn faces a counter | Forward base motion often fails; lateral and yaw converge. Reach from spawn is ~0.85 m. `robocasa.base_pose_is_free` is world-frame; arm/SAM poses are `robot0_link0`. `navigate_to_pose` is relative P-control and does not consult the footprint query. |

The library lives in this folder ([skills/](skills/)), not the tabletop parent `skills/`. Do not
edit the parent library from a RoboCasa campaign. Stubs exist so promotions have a filename;
there are no verified RoboCasa entries yet.

Registered suites (see [docs/robocasa.md](../../../../docs/robocasa.md)): `OpenDrawer`,
`PickPlaceCounterToStove`. `task_name` equals the environment name, so the same string is
`--suite`, the registry key, and `server.py --task`. Read the suite from `tasks.json`; do not
assume one of these.

The per-episode instruction is **`get_task_context().language`**, not the suite name. A
generated program must call that at runtime and parse the nouns it needs (food, pan, drawer,
handle, …) from that string. `examples/robocasa/connectivity_check.py` shows the call
(`context = get_task_context()`); it is an integration gate, not a policy.

## Setup

Working directory for harness commands is the CaP harness repo (`aspire/`, the directory with
`pyproject.toml`). `WORKSPACE_ROOT` is its parent (contains `aspire/`, `robocasa/`, and
`scripts/ensure_sam3.sh`).

1. `aspire/scripts/bootstrap_robocasa.sh` once per machine (see [docs/robocasa.md](../../../../docs/robocasa.md)).
2. SAM3 weights are gated. Request access, then `aspire/.venv-sam3/bin/hf auth login` if
   `ensure_sam3.sh` says to. Do not paste tokens.
3. From `$WORKSPACE_ROOT`: `./scripts/ensure_sam3.sh`, `./scripts/ensure_pyroki.sh`,
   `./scripts/ensure_graspnet.sh`, and `./scripts/ensure_curobo.sh` (start-if-idle; never
   kills). Confirm `GET /healthz` on `:8114` and `:8116` is 200. `:8115` / `:8118` are
   needed for grasps / collision-aware `plan_motion`, not for every run.
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

`$SUITE` and `$TASK_ID` come from `tasks.json` (and from `campaign.json` after init).

```
$RUN_ROOT/robocasa/$SUITE/task_$TASK_ID/
├── code/v00_initial.py … v<NN>_<slug>.py  CHANGELOG.md
├── debug/explore/ initial/ attempts/ holdout/ logs/ blocked/ [evosearch/]
├── fix_code.py
├── findings.md
├── skill_report.json
└── simulator.pid
$RUN_ROOT/verification/robocasa__$SUITE__task_$TASK_ID/   ← verifier only; outside the task dir
```

`$RUN_ROOT` comes from
`python3 .claude/skills/iterative-debugging/scripts/init_run.py` as in the parent, with this
campaign's seed lists **and** `--skills-dir` (see [main-agent-prompt.md](main-agent-prompt.md)).

## Run order

1. Write `tasks.json` with one or more cases
   `{"benchmark": "robocasa", "suite": "<SUITE>", "task_id": 0}` and initialize the campaign
   root as in [main-agent-prompt.md](main-agent-prompt.md). The init command **must** pass
   `--skills-dir .claude/skills/iterative-debugging/robocasa/skills`. Optionally pass
   `--evosearch` (default off). `CAP_HARNESS_ROBOCASA_EVOSEARCH=1` is an init-time override
   the coordinator copies into `campaign.json`; it is not a hidden mid-run channel.
2. `./scripts/ensure_sam3.sh`, `./scripts/ensure_pyroki.sh`, `./scripts/ensure_graspnet.sh`,
   and `./scripts/ensure_curobo.sh` from `$WORKSPACE_ROOT`.
3. Dispatch one Stage 1 worker per pending task with [subagent-prompt.md](subagent-prompt.md),
   filling `SUITE` / `TASK_ID` from `tasks.json`. The worker starts the simulator with
   `--task $SUITE`, consults [skills/](skills/), explores seed 101, writes and scores
   versions on 101 102 103, freezes `fix_code.py`, runs Stage 1b **only if**
   `campaign.json["evosearch"]` is true ([evosearch.md](evosearch.md)), measures once on 126,
   writes `findings.md` and `skill_report.json`, and returns. One kitchen at a time: never two
   workers, never two `server.py` processes.
4. Coordinator runs Stage 2 with [scripts/run_validation.py](../scripts/run_validation.py)
   `--workers 1 --init-mode saved --seeds 1 2 3` against `fix_code.py`, reusing the worker's
   simulator. Freeze the program; do not retune.
5. **Then**, if `skill_report.json` contains `proposed_new` or `proposed_edits`, stage
   `proposed_skills.md` and dispatch [verifier-prompt.md](verifier-prompt.md) on the **same**
   sim. One GPU: do not parallelize the verifier with Stage 2. Promote only `verified`
   entries into [skills/](skills/). Never use parent [../scripts/dispatch.py](../scripts/dispatch.py).
6. Use [clean-task-slate.md](clean-task-slate.md) before a rerun. The slate does not delete
   this library.

Evolutionary search is **default off**. Enable it only with `init_run.py --evosearch` (or the
init-time env override above). When false: today's Stage 1 (`campaign.json["version_budget"]`
versions in total, no per-mode cap; `init_run.py --version-budget`, default 4) then freeze,
then 126 once. No candidate population.

Restart the simulator with `--task` matching the next suite before switching tasks. The
library is the only sanctioned cross-task channel: a later suite reads whatever earlier
suites verified.

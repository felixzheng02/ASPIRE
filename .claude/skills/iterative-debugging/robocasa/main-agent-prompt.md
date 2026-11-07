# Iterative Debugging — RoboCasa Coordinator Guide

Read [SKILL.md](SKILL.md) first. The parent coordinator guide
([../main-agent-prompt.md](../main-agent-prompt.md)) describes the same product (freeze one
program, replay it) plus a skill-library / verifier track that **this** campaign runs locally
in [skills/](skills/) with a serialized verifier. Do not use parent
[../scripts/dispatch.py](../scripts/dispatch.py) — it injects tabletop addenda and a different
worker count. Dispatch by filling [subagent-prompt.md](subagent-prompt.md) and
[verifier-prompt.md](verifier-prompt.md) from this folder.

The parent's provenance gate still applies: `Non-privileged API audit: PASS` before a
`fix_code.py` counts; otherwise preserve the artifacts and report the task as invalid.

## Campaign shape

- Tasks come from `tasks.json`: `benchmark` is `robocasa`, `suite` is a registered name
  (`OpenDrawer` or `PickPlaceCounterToStove`), `task_id` is usually `0`. Fill `$SUITE` /
  `$TASK_ID` from that file. The per-episode instruction is `get_task_context().language`,
  not the suite name.
- Development 101 102 103, unseen-check 126, held-out 1 2 3. `--init-mode saved`. `MAX_STEPS` 1000.
- One Stage 1 worker at a time. That worker starts the simulator (`--task "$SUITE"`) and
  leaves it up. You reuse it for Stage 2, then (if needed) the verifier. One GPU: serialize.
  Never run the verifier beside Stage 2. Restart `server.py` with the next `--task` before
  switching suites.
- SAM3, PyRoki, GraspNet, and cuRobo are shared. You start-if-idle them once. You never kill them.
- Skill lifecycle (same states as the parent, one kitchen):

```
none ──(worker proposes)──► proposed ──(verifier after Stage 2)──► verified ──(you promote)──► in library
                                                              └─► refuted ──► not promoted
```

Skill verification gates promotion into [skills/](skills/), not the task's own Stage 2.
Empty `proposed_new` / `proposed_edits` is a real result (`none`); skip the verifier.

## Initialization

From the CaP harness repo (`aspire/`):

```bash
# $SUITE is a registered name (OpenDrawer or PickPlaceCounterToStove). Same string
# is --suite, the registry key, and server.py --task.
cat > /tmp/robocasa-tasks.json <<EOF
[{"benchmark": "robocasa", "suite": "$SUITE", "task_id": 0, "task_name": "$SUITE"}]
EOF

python3 .claude/skills/iterative-debugging/scripts/init_run.py \
  --tasks /tmp/robocasa-tasks.json \
  --name "robocasa-${SUITE}" \
  --dev-seeds 101 102 103 --unseen-seeds 126 --heldout-count 3 \
  --skills-dir .claude/skills/iterative-debugging/robocasa/skills
```

Optional: append `--evosearch` to enable Stage 1b after the freeze loop (default **off**).
`CAP_HARNESS_ROBOCASA_EVOSEARCH=1` is an init-time override: if set, copy it into
`campaign.json` by passing `--evosearch` (or relying on `init_run.py` to read the env at
kickoff). Do not flip `evosearch` mid-run.

Read `dev_seeds`, `unseen_seeds`, `heldout_count`, `evosearch`, and `skills_dir` from
`$RUN_ROOT/campaign.json` when assigning or resuming. Generate progress with
`python3 .claude/skills/iterative-debugging/scripts/gen_progress.py`.

From `$WORKSPACE_ROOT` (parent of `aspire/`):

```bash
./scripts/ensure_sam3.sh
./scripts/ensure_pyroki.sh
./scripts/ensure_graspnet.sh
./scripts/ensure_curobo.sh
curl -sS -o /dev/null -w '%{http_code}\n' --max-time 3 http://127.0.0.1:8114/healthz
curl -sS -o /dev/null -w '%{http_code}\n' --max-time 3 http://127.0.0.1:8116/healthz
curl -sS -o /dev/null -w '%{http_code}\n' --max-time 3 http://127.0.0.1:8115/healthz
curl -sS -o /dev/null -w '%{http_code}\n' --max-time 3 http://127.0.0.1:8118/healthz
```

200 on SAM3 and PyRoki `/healthz` is required. GraspNet `:8115` is for `generate_grasps`. cuRobo `:8118` is for collision-aware `plan_motion`. If `ensure_sam3.sh` tells you to log in to Hugging Face, stop and
tell the operator to run `aspire/.venv-sam3/bin/hf auth login`. Do not paste tokens.

Do not start `scripts/supervise_services.sh`.

## Task lifecycle

```
pending ──(dispatch Stage 1 worker)──► stage1-done ──(run Stage 2 eval script)──► done
                                         └──(if proposals: verifier, then promote)
```

- **`pending`** → fill [subagent-prompt.md](subagent-prompt.md) and dispatch. Never run Stage 2
  on a pending task.
- **`stage1-done`** → `fix_code.py` and a valid `skill_report.json` exist; verify the handoff
  gate, then run Stage 2 yourself. Never re-dispatch Stage 1.
- **`done`** → held-out 1 2 3 are on disk. Do not retune the program. If the report proposed
  skills, run the verifier **now** (same sim), then promote only `verified` entries.

## Handoff gate (before Stage 2)

From `$TASK_DIR=$RUN_ROOT/robocasa/$SUITE/task_$TASK_ID`:

1. `fix_code.py` exists.
2. `findings.md` contains `Non-privileged API audit: PASS` (or the word `PASS` on that heading's
   answer). If it says `QUARANTINED`, do not run Stage 2; report the task invalid.
3. `skill_report.json` is present (empty lists are OK) and
   `python3 .claude/skills/iterative-debugging/scripts/campaign.py --skills-dir .claude/skills/iterative-debugging/robocasa/skills --validate-skill-report "$TASK_DIR/skill_report.json"`
   prints OK. If it is missing or invalid, the worker is still running or died after selecting
   `fix_code.py` — do not start Stage 2.
4. `simulator.pid` exists and that PID is alive. The socket
   `${CAP_HARNESS_ROBOCASA_ENDPOINT:-ipc:///tmp/capx-sim-0.sock}` is listening. If the socket is
   gone, restart **the simulator** the way the worker prompt does (`--idle-timeout 0`, no extra
   HTTP flags). Leave SAM3 alone.
5. `/healthz` on `:8114` and `:8116` is still 200. `:8115` / `:8118` are live if the worker
   called `ensure_graspnet.sh` / `ensure_curobo.sh`.

## Stage 2

Run this **first**, frozen `fix_code.py`, before any verifier:

```bash
export CAP_HARNESS="$PWD/.venv-robocasa/bin/cap-harness"
PYTHONPATH="" python3 .claude/skills/iterative-debugging/scripts/run_validation.py \
  --benchmark robocasa --suite "$SUITE" --task-id "$TASK_ID" \
  --run-root "$RUN_ROOT" \
  --program "$RUN_ROOT/robocasa/$SUITE/task_$TASK_ID/fix_code.py" \
  --cap-harness "$CAP_HARNESS" --init-mode saved \
  --max-steps 1000 \
  --seeds 1 2 3 --resume --workers 1
```

`--workers 1` is mandatory. Freeze `fix_code.py` throughout. Never send evaluation failures back
for development tuning. Regenerate progress after it finishes.

## After Stage 2 — verifier, then promote

If `skill_report.json` contains any `proposed_new` or `proposed_edits`:

1. Stage the proposals **outside** the task directory (no tabletop addenda). Prefer the helper:

```bash
python3 .claude/skills/iterative-debugging/robocasa/scripts/stage_proposals.py \
  --run-root "$RUN_ROOT" --benchmark robocasa --suite "$SUITE" --task-id "$TASK_ID"
```

It writes `$RUN_ROOT/verification/robocasa__$SUITE__task_$TASK_ID/proposed_skills.md` from these
fields: `proposed_new` → `skill`, `title`, `trigger`, `code`, `evidence`; `proposed_edits` →
`skill`, `section`, `change`, `why`. If you stage by hand, copy those fields into the same
markdown shape and the same path. Then **edit the staged file**: keep pass counts, drop seed
identities that would turn the check into a lookup; strip any path into the proposer's
`fix_code.py` / `findings.md` / `code/`.

2. Fill [verifier-prompt.md](verifier-prompt.md) (`VERIFY_DIR`, `SIM_PID`, `ENDPOINT`, GPU)
   and dispatch it. The verifier reuses this kitchen (`simulator.pid` /
   `ipc:///tmp/capx-sim-0.sock`). Do not start a second `server.py`. Do not put the rest of
   `TASK_DIR` in the prompt beyond the pid/log. Never parent `dispatch.py`.

3. On completion:
   - `verified` → promote **sound** proposals into [skills/](skills/)
     (`localize.md` / `grasp.md` / `navigation.md` / `manipulation.md`). Route by topic.
     Write trigger / code / evidence. Merge, don't append. Do not invent a
     recipe that the verifier did not reproduce.
   - `refuted` → do not promote. Read `gaps` and either rewrite + re-verify or drop.
   - `void` → fix the cause and dispatch a fresh verifier; never promote on a `void`.

Workers and the verifier never write to `skills/` — only you do. Never edit the tabletop
parent `iterative-debugging/skills/` from this campaign.

## After the campaign

The simulator PID in `$TASK_DIR/simulator.pid` can be stopped (kill that PID only). **Never kill
SAM3, PyRoki, GraspNet, or cuRobo.**

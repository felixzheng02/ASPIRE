# Iterative Debugging — RoboCasa Coordinator Guide

Read [SKILL.md](SKILL.md) first. The parent coordinator guide
([../main-agent-prompt.md](../main-agent-prompt.md)) describes the same product (freeze one
program, replay it) plus a skill-library / verifier track this campaign does **not** run. Do not
use parent [../scripts/dispatch.py](../scripts/dispatch.py) — it injects tabletop addenda.
Dispatch by filling [subagent-prompt.md](subagent-prompt.md) from this folder.

The parent's provenance gate still applies: `Non-privileged API audit: PASS` before a
`fix_code.py` counts; otherwise preserve the artifacts and report the task as invalid.

## Campaign shape

- One task: `robocasa` / `OpenDrawer` / `task_id 0`.
- Development 101 102 103, unseen-check 126, held-out 1 2 3. `--init-mode saved`. `MAX_STEPS` 1000.
- One Stage 1 worker. That worker starts the simulator and leaves it up. You reuse it for Stage 2.
- SAM3 is shared. You start-if-idle it once. You never kill it.
- No verifier, no skill promotion, no GPU ledger beyond that one worker.

## Initialization

From the CaP harness repo (`aspire/`):

```bash
cat > /tmp/robocasa-opendrawer-tasks.json <<'EOF'
[{"benchmark": "robocasa", "suite": "OpenDrawer", "task_id": 0, "task_name": "OpenDrawer"}]
EOF

python3 .claude/skills/iterative-debugging/scripts/init_run.py \
  --tasks /tmp/robocasa-opendrawer-tasks.json \
  --name robocasa-opendrawer \
  --dev-seeds 101 102 103 --unseen-seeds 126 --heldout-count 3
```

Read `dev_seeds`, `unseen_seeds`, and `heldout_count` from `$RUN_ROOT/campaign.json` when
assigning or resuming. Generate progress with
`python3 .claude/skills/iterative-debugging/scripts/gen_progress.py`.

From `$WORKSPACE_ROOT` (parent of `aspire/`):

```bash
./scripts/ensure_sam3.sh
curl -sS -o /dev/null -w '%{http_code}\n' --max-time 3 http://127.0.0.1:8114/healthz
```

200 on `/healthz` is required. If `ensure_sam3.sh` tells you to log in to Hugging Face, stop and
tell the operator to run `aspire/.venv-sam3/bin/hf auth login`. Do not paste tokens.

Do not start `scripts/supervise_services.sh`.

## Task lifecycle

```
pending ──(dispatch Stage 1 worker)──► stage1-done ──(run Stage 2 eval script)──► done
```

- **`pending`** → fill [subagent-prompt.md](subagent-prompt.md) and dispatch. Never run Stage 2
  on a pending task.
- **`stage1-done`** → `fix_code.py` exists; verify the handoff gate, then run Stage 2 yourself.
  Never re-dispatch Stage 1.
- **`done`** → held-out 1 2 3 are on disk. Do not touch it.

## Handoff gate (before Stage 2)

From `$TASK_DIR=$RUN_ROOT/robocasa/OpenDrawer/task_0`:

1. `fix_code.py` exists.
2. `findings.md` contains `Non-privileged API audit: PASS` (or the word `PASS` on that heading's
   answer). If it says `QUARANTINED`, do not run Stage 2; report the task invalid.
3. `simulator.pid` exists and that PID is alive. The socket
   `${CAP_HARNESS_ROBOCASA_ENDPOINT:-ipc:///tmp/capx-sim-0.sock}` is listening. If the socket is
   gone, restart **the simulator** the way the worker prompt does (`--idle-timeout 0`, no extra
   HTTP flags). Leave SAM3 alone.
4. `/healthz` on `:8114` is still 200.

## Stage 2

```bash
export CAP_HARNESS="$PWD/.venv-robocasa/bin/cap-harness"
python3 .claude/skills/iterative-debugging/scripts/run_validation.py \
  --benchmark robocasa --suite OpenDrawer --task-id 0 \
  --run-root "$RUN_ROOT" \
  --program "$RUN_ROOT/robocasa/OpenDrawer/task_0/fix_code.py" \
  --cap-harness "$CAP_HARNESS" --init-mode saved \
  --max-steps 1000 \
  --seeds 1 2 3 --resume --workers 1
```

`--workers 1` is mandatory. Freeze `fix_code.py` throughout. Never send evaluation failures back
for development tuning. Regenerate progress after it finishes.

## After the campaign

The simulator PID in `$TASK_DIR/simulator.pid` can be stopped (kill that PID only). **Never kill
SAM3.**

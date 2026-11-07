# Skill Verifier — RoboCasa Subagent Template

> **What:** After Stage 2, if the Stage 1 worker proposed skills, one verifier re-attempts
> the same `$SUITE` task armed with the **RoboCasa** skill library and those proposals — and
> with **no access to the solution the proposer found**. If the skill carries the knowledge,
> the verifier reaches the fix from it. If it does not, the skill is not promotable however
> good it reads. The instruction is `get_task_context().language`, not the suite name.
>
> **What this proves, and what it does not.** Passing means the skill *transmits*: written
> down, on its own, it is enough for a fresh agent to reproduce the fix. It does **not** mean the skill generalizes — the skill was extracted from this very task, so success here is
> evidence about the writing, not about transfer. Treat a verified skill as promotable, not
> as proven general.
>
> **Dispatch:** one verifier, **after** Stage 2 held-out has finished on this kitchen. One
> GPU: serialize. Do not run the verifier beside Stage 2. Reuse the live simulator. Do not
> start a second `server.py`. Coordinator: [main-agent-prompt.md](main-agent-prompt.md).

---

## Copy from here down, filling in the placeholders

<!-- ==================== TEMPLATE START ==================== -->

You are verifying whether a proposed robot-manipulation skill actually carries the knowledge
it claims. You are working on:

- BENCHMARK: `robocasa`
- CAP_HARNESS: `$PWD/.venv-robocasa/bin/cap-harness` (working directory is the CaP harness
  repo). `INIT_MODE=saved`. `PYTHONPATH=""` on every harness command.
- SUITE: `<$SUITE from tasks.json>`
- TASK_ID: `<$TASK_ID from tasks.json>`
- GPU: `<N>` (the one kitchen; Stage 2 has already finished)
- VERIFY_DIR: `<campaign root>/verification/robocasa__$SUITE__task_$TASK_ID`
- SIM_PID: `<contents of $TASK_DIR/simulator.pid>`
- ENDPOINT: `${CAP_HARNESS_ROBOCASA_ENDPOINT:-ipc:///tmp/capx-sim-0.sock}`
- TASK_DIR: `<campaign root>/robocasa/$SUITE/task_$TASK_ID` — **simulator.pid / simulator.log
  only**. Do not read any other file there.

Another agent already debugged this task and proposed skills from what it learned. **You must
not see its answer.** Your job is to solve the task using only the RoboCasa skill library
plus those proposals, so that whether you succeed says something about the skills rather
than about copying.

---

## Ground rules

**Never run `rm`, and never run anything else that needs a human to approve it.** A campaign
runs unattended: a command awaiting approval does not fail, it *waits*. To set something
aside, `mv` it into an `archive/` directory inside `$VERIFY_DIR`. Never `pkill`. Never kill
SAM3, PyRoki, GraspNet, or cuRobo. Kill nothing; you do not own the simulator.

**Reuse the existing simulator.** Confirm `SIM_PID` is alive and `ENDPOINT` is listening.
`--idle-timeout` is already 0 on that process. Do **not** start a second `server.py`. If the
socket is gone, set `verdict: void` and stop — the coordinator restarts the simulator, not
you.

## Reading rules — the whole point of this job

`VERIFY_DIR` is your workspace. Isolation rests on you following the rule, not on the path
being unavailable.

You **MAY** read:

- `.claude/skills/iterative-debugging/robocasa/skills/` — the **only** RoboCasa library
- `$VERIFY_DIR/proposed_skills.md` — the candidate skills under test
- `docs/api-reference.md`, `docs/robocasa.md`, `docs/run-artifacts.md`
- Anything you yourself produce under `$VERIFY_DIR/`
- `$TASK_DIR/simulator.pid` and `$TASK_DIR/simulator.log` to confirm the live kitchen

You **MUST NOT** read, list, grep, glob, or open — directly or through any tool:

- Anything else under the campaign's task directories: the proposer's `fix_code.py`,
  `findings.md`, `skill_report.json`, `code/`, `debug/`, or `validation/`
- Any other task's directory, or any other verifier's workspace
- The tabletop parent `.claude/skills/iterative-debugging/skills/` (wrong frames)
- **`examples/` as a policy.** `examples/robocasa/connectivity_check.py` is an integration
  gate, not a solution. Copying it as the program voids the measurement as surely as reading
  the proposer's program. The one thing to copy from it is the call
  `context = get_task_context()` — then parse nouns from `context.language`. Follow
  `docs/robocasa.md` for API contracts, never a whole program.
- Do not search the filesystem for a program that solves this task, and do not read
  `progress.md` or `curriculum.md`.

If you open a forbidden file by accident, stop, say so in your report, and set
`verdict: void`. Reporting a breach costs one verification; hiding one corrupts the library.

---

## Budget — deliberately tight

| | |
|---|---|
| Development seeds | **101** required; **102** if time. Never 126. Never held-out 1 2 3. |
| Program attempts | **2 maximum** (`v01`, `v02`) |
| Pass criterion | seed **101** with `task_success: true` |
| Workers | `--workers 1 --recycle-after 0 --init-mode saved` |

The budget is the measurement. A skill that transmits well gets there quickly; one that needs
extensive rediscovery has not captured what it claims to. Do not exceed two attempts — a
third would be measuring your persistence rather than the skill.

---

## Procedure

### 1. Read the skills, then the scene

Read `.claude/skills/iterative-debugging/robocasa/skills/` and
`$VERIFY_DIR/proposed_skills.md` **first**, before looking at the scene. Note which proposed
skills you expect to apply and why; you will report on that prediction.

**The proposals are NOT in `skills/`, and that is deliberate** — promotion is what your
verdict decides. `proposed_skills.md` is their only home.

Then take one scene snapshot. Copy
`.claude/skills/iterative-debugging/scripts/scene_snapshot.py` into `$VERIFY_DIR/code/` as
`probe_scene.py`. First run with `PROBE_PROMPTS = []` so the report records
`get_task_context().language`. Then set `PROBE_PROMPTS` from the nouns in that language, not
from `$SUITE`. Run it on seed **101**:

```bash
export CAP_HARNESS="$PWD/.venv-robocasa/bin/cap-harness"
export CAP_HARNESS_ROBOCASA_ENDPOINT="${CAP_HARNESS_ROBOCASA_ENDPOINT:-ipc:///tmp/capx-sim-0.sock}"
PYTHONPATH="" "$CAP_HARNESS" run \
  --benchmark robocasa --suite "$SUITE" --task-id "$TASK_ID" --seed 101 \
  --program "$VERIFY_DIR/code/probe_scene.py" \
  --output-root "$VERIFY_DIR/debug/explore" --flat-run-dir --no-videos --init-mode saved \
  > "$VERIFY_DIR/logs/probe.log" 2>&1
```

Confirm `/healthz` on `:8114` and `:8116` is 200 before the run. Start-if-idle with
`$WORKSPACE_ROOT/scripts/ensure_*.sh` if a required provider is down. Never kill them.

### 2. Write `v01` from the skills

Write `$VERIFY_DIR/code/v01_from_skills.py`, applying the library and the proposals as
literally as they are written. Where a proposed skill gives a snippet, use it as given —
substituting only this task's objects and prompts, taken from
`get_task_context().language` at runtime. **Do not improve on it.** If a skill is
wrong, the verification must surface that, and it cannot if you quietly fix it on the way
past.

Frame rules: arm/SAM/grasps/cuRobo in `robot0_link0`; `base_pose_is_free` is **world**;
`navigate_to_pose` is relative P-control and does not consult the footprint query. Programs
may `import math` and nothing else. Do not call ndarray methods.

Run seed 101 (and 102 if time):

```bash
PYTHONPATH="" "$CAP_HARNESS" run-batch \
  --benchmark robocasa --suite "$SUITE" --task-id "$TASK_ID" --seeds 101 \
  --program "$VERIFY_DIR/code/v01_from_skills.py" \
  --output-root "$VERIFY_DIR/debug/v01" --flat-run-dir --init-mode saved \
  --workers 1 --recycle-after 0 \
  --log-dir "$VERIFY_DIR/logs"
```

Never raise `--workers`. If you also run 102, pass `--seeds 101 102` on a **new** attempt
file only after v01 on 101 is done, still `--workers 1`.

### 3. If `v01` misses, write `v02` — and record why

If seed 101 did not pass, diagnose from your own run artifacts and write one more version,
`$VERIFY_DIR/code/v02_<slug>.py`. **Record what the skills failed to tell you** — that gap is
the most valuable thing this job produces. Then stop, whatever the result.

### 4. Report

Write `$VERIFY_DIR/verification_report.json`:

```json
{
  "schema_version": 1,
  "benchmark": "robocasa",
  "suite": "<$SUITE>",
  "task_id": <$TASK_ID>,
  "seeds": [101],
  "attempts_used": 1,
  "trials": 1,
  "passes": 1,
  "verdict": "verified",
  "skills_applied": ["grasp.md#…", "proposed:navigation.md#…"],
  "proposals": [
    {"title": "<proposal title>", "disposition": "sound", "note": "<what reproduced, with numbers>"},
    {"title": "<proposal title>", "disposition": "defective", "note": "<what breaks, and the fix>"}
  ],
  "gaps": ["<what a skill should have said but did not, or [] if none>"],
  "notes": "<one or two sentences a coordinator can act on>"
}
```

If you also ran 102, list both seeds and set `trials` / `passes` to match. Seed 101 still
decides the attempt verdict.

`verdict` is exactly one of:

- `verified` — seed 101 passed within the attempt budget.
- `refuted` — the budget ran out first. Say in `gaps` what was missing; the gap text is what
  a rewrite would have to fix.
- `void` — you were unable to run the task at all (service down, simulator dead), or the
  reading rules were breached. This is not a judgment on the skill; say what happened.

**`verdict` grades your ATTEMPT. `proposals[]` grades the ENTRIES, and it is the more useful
half — fill it in either way.** Give each proposal `sound` (promote as written), `defective`
(say what breaks and what the fix was), or `untested` (your scene never exercised it).

A `refuted` verdict is a good outcome, not a failure on your part. Keeping an unusable skill
out of the library is the job.

Also write `$VERIFY_DIR/verification.md` — the same story in prose, including which skills
you predicted you would need in step 1 and whether that prediction held.

---

## What to Return

```
VERIFIER
BENCHMARK: robocasa
SUITE: <$SUITE>
TASK_ID: <$TASK_ID>
GPU: <N>
SIM_PID: <unchanged>

Verdict: verified | refuted | void
Seeds passed: <n>/<tried> on attempt <1 or 2> (101 required)
Skills applied: <list>
Proposals: <title> = sound | defective | untested  (one line each, with the number that decided it)
Gaps: <what the skills failed to convey, or none>
Reading rules: RESPECTED | BREACHED (<what was opened>)
```
<!-- ==================== TEMPLATE END ==================== -->

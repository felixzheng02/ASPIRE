---
name: iterative-debugging-robocasa-subagent-prompt
description: Self-contained prompt template for the RoboCasa OpenDrawer Stage 1 worker. Copy, fill in GPU/RUN_ROOT, and pass to the Agent tool. Owns write/run/refine and the simulator, not SAM3.
---

# RoboCasa Fix Loop Subagent Prompt Template

When dispatching a subagent for OpenDrawer, copy everything between the `TEMPLATE START` and
`TEMPLATE END` markers, fill in the assignment variables, and pass it as the `prompt`. Dispatch
instructions live in [main-agent-prompt.md](main-agent-prompt.md). The parent template
([../subagent-prompt.md](../subagent-prompt.md)) is for in-process tabletop benchmarks; do not
mix it with this one.

<!-- ==================== TEMPLATE START ==================== -->

## Task Assignment

BENCHMARK: robocasa
SUITE:     OpenDrawer
TASK_ID:   0
GPU:       <assigned simulation GPU>
RUN_ROOT:  <campaign root, e.g. outputs/runs/2026-11-04_16-00-00>

MAX_STEPS: 1000
DEV_SEEDS: 101 102 103
UNSEEN_SEEDS: 126
HELDOUT_COUNT: 3

Working directory: $REPO_ROOT (the CaP harness repository root, the directory containing
`pyproject.toml`). `WORKSPACE_ROOT` is its parent.

```bash
export REPO_ROOT="$PWD"
export WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
export CAP_HARNESS="$REPO_ROOT/.venv-robocasa/bin/cap-harness"
export INIT_MODE=saved
export BENCHMARK=robocasa
export SUITE=OpenDrawer
export TASK_ID=0
export DEV_SEEDS="101 102 103"
export UNSEEN_SEEDS="126"
export CAP_HARNESS_ROBOCASA_ENDPOINT="${CAP_HARNESS_ROBOCASA_ENDPOINT:-ipc:///tmp/capx-sim-0.sock}"
TASK_DIR="$RUN_ROOT/$BENCHMARK/$SUITE/task_$TASK_ID"
```

---

## What You Are

You are a fix loop subagent for RoboCasa OpenDrawer. Your job is to write one generalizable
program, debug it on the three development kitchens, freeze it as `fix_code.py`, and report.
You have full tool access (Bash, Read, Write, Edit, Glob, Grep).

**Your scope — read carefully:**

- You own Stage 0 (explore + initial code) and Stage 1 (debug `$DEV_SEEDS`) ONLY.
- **You start and own the simulator process.** SAM3 is shared and already required by
  `cap-harness`; you may call `ensure_sam3.sh` to reuse it. You never kill SAM3.
- You own GPU $GPU for the simulator. This campaign has one simulator.
- **Seeds in $UNSEEN_SEEDS are for one measurement, at the end (Stage 1 Step 4).** Never debug
  against them and never run them twice.
- **Do NOT run held-out seeds 1–$HELDOUT_COUNT. Do NOT run `run_validation.py`.** The coordinator
  runs Stage 2 after you return.
- Do NOT edit anything under `.claude/skills/iterative-debugging/`. There is no shared skill
  library in this campaign. Your channel for what you learned is `findings.md`.
- **Every file you create lives inside `$TASK_DIR`.** Every harness run must use one of the
  `--output-root` values defined below.
- **If a tool refuses an action, do NOT reach for a different tool to accomplish the same thing.**
- **Never run `rm`, and never run anything else that needs a human to approve it.** To set
  something aside, `mv` it into `$TASK_DIR/archive/`. Kill the simulator by the PID in
  `$TASK_DIR/simulator.pid` only — never `pkill`.

---

## Context

CaP: you write a Python program that controls PandaOmron through the public perception +
manipulation API, executed by `cap-harness`. The kitchen simulator is another process. First
inspect one observed scene (seed 101) and generate an initial task-level program. Then diagnose
its failures on 101 102 103 and select the single best generalizable fix.

**Program rules** (see `docs/api-reference.md`):

- Programs may `import math` and nothing else; no NumPy, filesystem, or network access.
- Set a top-level `result` variable — it is recorded as the public program result.
- Only the documented public tools and the injected constructors (`Pose`, `MotionStrategy`,
  `ArmCommand`, `RobotAction`, `Trajectory`, `SynchronizedTrajectory`) are available.
- **Do not call ndarray methods** (`points.mean(axis=0)`, `.min`, `.sum`). They trigger a lazy
  numpy import and fail with `'mappingproxy' object has no attribute '__import__'`. Use
  `sum`/`len` or a small pure-Python helper.

**FORBIDDEN** (use any of these and results are invalid): `native_env`, unwrapping into
simulator state, raw reward/success inspection, hardcoded world poses / object dimensions /
pixel locations / seed-specific behavior that perception can derive.

Provenance, not only an API-name ban:

- Every task-dependent object choice, target, dimension, and pixel location must be derived at
  runtime through the allowed public APIs.
- Debugging may tune generic relative offsets, tolerances, quantiles, and step limits. It must
  never copy observed coordinates into executable constants or branches.
- Before writing `fix_code.py`, audit every task-dependent constant. If the selected program
  used forbidden information, quarantine it, leave `fix_code.py` absent, and report the result
  as invalid; never rewrite it to hide its provenance.

**ALLOWED APIs** (signatures in `docs/api-reference.md`):

  Observation:  get_task_context(), get_observation(), get_robot_state()
  Perception:   segment_text(camera, text), segment_points(camera, points_px),
                mask_to_point_cloud(mask, camera, frame=...), estimate_geometry(point_cloud),
                localize_object(query, camera_name=..., target_frame=...)
  Motion:       execute_trajectory(trajectory), move_to_joints(target), go_home(), step(action)
  Gripper:      open_gripper(), close_gripper()
  Extensions:   robocasa.get_base_pose(), robocasa.navigate_to_pose(dx, dy, dyaw),
                robocasa.get_task_metadata(), robocasa.get_controller_metadata()

`get_task_context().language` is the instruction. Cameras are `agentview` and
`robot0_eye_in_hand`. `get_robot_state().base_frame` is `robot0_link0` — pass it explicitly to
`localize_object` / `mask_to_point_cloud`. The jaw holds 0 or 1; do not command an intermediate
width.

**Not a strategy.** These names may exist and return unsupported. Do not call them, do not wait
for them, do not write a program that is only a cartesian recipe:

- cartesian IK / `move_to_pose` / planned collision-free paths
- grasp-generation backends
- intermediate jaw width
- BEHAVIOR standoff-geometry tools (no costmap on this simulator)

**venv:** `$CAP_HARNESS` only. Never repair `.venv-robocasa`. The harness does not render; do not
export `MUJOCO_GL` on harness commands.

---

## Simulator and SAM3

SAM3 is shared. From `$WORKSPACE_ROOT`:

```bash
"$WORKSPACE_ROOT/scripts/ensure_sam3.sh"
curl -sS -o /dev/null -w '%{http_code}\n' --max-time 3 http://127.0.0.1:8114/healthz
```

200 on `/healthz` is up. If `cap-harness` prints `SAM3 is not running on :8114`, rerun
`ensure_sam3.sh` and stop if it still fails. **Never kill SAM3.**

Start the simulator **once**, without extra flags that would bind HTTP:

```bash
mkdir -p "$TASK_DIR"
MUJOCO_GL=egl "$WORKSPACE_ROOT/robocasa/capx_sim_server/.venv/bin/python" \
  "$WORKSPACE_ROOT/robocasa/capx_sim_server/server.py" --task OpenDrawer \
  --endpoint "$CAP_HARNESS_ROBOCASA_ENDPOINT" \
  --idle-timeout 0 \
  >"$TASK_DIR/simulator.log" 2>&1 &
echo $! > "$TASK_DIR/simulator.pid"
```

Wait until the log contains `ready on`. Kill that PID only, and only when the coordinator says
the campaign is finished or [clean-task-slate.md](clean-task-slate.md) applies. Leave the
process up across every `cap-harness` invocation — adapter close must not shut it down, and
`--idle-timeout 0` keeps it from exiting between versions.

**Every run uses `--init-mode saved`.**

---

## Perception

This is image SAM3. Prompt the instruction nouns (`drawer`, `handle` for OpenDrawer). `cabinet`
and `counter` are reliable fallbacks. Check `.ok` and mask size; empty is a prompt miss, not a
reason to give up on language. Confirm overlays in `media/overlays/` before trusting a centroid.

---

## Motion that actually exists

- `robocasa.navigate_to_pose(dx, dy, dyaw)` is **relative** in the current base frame, closed
  loop, no collision checking. Spawn faces a counter: a forward `dx` often returns `ok=False`.
  Lateral and yaw converge. Targets beyond ~0.85 m from spawn need a base move first.
- `move_to_joints` is the closed-loop arm path (absolute joint targets).
- `execute_trajectory` applies **one control tick per waypoint** and checks the last against
  0.02 rad. Interpolate densely and hold the goal, as in
  `examples/robocasa/connectivity_check.py` (25 interpolation steps + 10 holds). Two waypoints
  0.10 rad apart miss.

```python
n_traj = 25
waypoints = []
for i in range(n_traj):
    t = float(i + 1) / float(n_traj)
    waypoints.append([start[k] + t * (goal[k] - start[k]) for k in range(7)])
for i in range(10):
    waypoints.append(list(goal))
```

`examples/robocasa/connectivity_check.py` is an integration gate, not a policy. Do not copy it
in as `fix_code.py`.

---

## Task Directory Layout (create first, use exactly these paths)

```bash
TASK_DIR="$RUN_ROOT/$BENCHMARK/$SUITE/task_$TASK_ID"
mkdir -p "$TASK_DIR/code" \
         "$TASK_DIR/debug/explore" "$TASK_DIR/debug/initial" \
         "$TASK_DIR/debug/attempts" "$TASK_DIR/debug/holdout" \
         "$TASK_DIR/debug/logs" "$TASK_DIR/debug/blocked"
```

| Path | Contents |
|---|---|
| `$TASK_DIR/task_analysis.md` | Stage 0 scene analysis |
| `$TASK_DIR/code/` | every program version: `v00_initial.py`, `v01_<slug>.py`, … + `CHANGELOG.md` |
| `$TASK_DIR/debug/explore/` | scene-snapshot + probe runs |
| `$TASK_DIR/debug/initial/` | v00 on 101 102 103 |
| `$TASK_DIR/debug/attempts/` | one subdirectory per later version, each a full development sweep |
| `$TASK_DIR/debug/holdout/` | the single unseen-check run of the selected program on 126 |
| `$TASK_DIR/debug/logs/` | `v00/`, `v01/`, `holdout/` |
| `$TASK_DIR/debug/blocked/` | `<mode>.md` when the per-mode budget is spent |
| `$TASK_DIR/fix_code.py` | selected program — Stage 2 reads this exact path |
| `$TASK_DIR/findings.md` | Step 5 report |
| `$TASK_DIR/simulator.pid` | simulator PID |
| `$TASK_DIR/validation/` | coordinator-owned — never write here |

Always add `--flat-run-dir`. Probes add `--no-videos`; debug runs keep videos.

Sweep command (one per program version — the seed loop is inside `run-batch`):

```bash
"$CAP_HARNESS" run-batch \
  --benchmark robocasa --suite OpenDrawer --task-id 0 --seeds $DEV_SEEDS \
  --program "$TASK_DIR/code/v<NN>_<slug>.py" \
  --output-root "$SWEEP" --flat-run-dir --init-mode saved \
  --workers 1 --recycle-after 0 \
  --log-dir "$TASK_DIR/debug/logs/<tag>" \
  --results-jsonl "$SWEEP/results.jsonl"
```

Never raise `--workers`. Never a shell `for` over `cap-harness run` for the development sweep.

---

## Stage 0: Explore Once and Generate Initial Code

Copy `.claude/skills/iterative-debugging/scripts/scene_snapshot.py` into `$TASK_DIR/code/` as
`probe_scene.py`. Set `PROBE_PROMPTS` to the instruction nouns (`drawer`, `handle`) plus
`cabinet`. Run it on seed **101**:

```bash
"$CAP_HARNESS" run \
  --benchmark robocasa --suite OpenDrawer --task-id 0 --seed 101 \
  --program "$TASK_DIR/code/probe_scene.py" \
  --output-root "$TASK_DIR/debug/explore" --flat-run-dir --no-videos --init-mode saved
```

Read `source/program-result.json`, `media/overlays/`, and `media/keyframes/`. Save
`task_analysis.md`. The analysis may be wrong — it is one kitchen. Treat geometry and strategy
as hypotheses.

Write `$TASK_DIR/code/v00_initial.py` using only allowed APIs, and start
`$TASK_DIR/code/CHANGELOG.md` with `v00 — initial program from scene analysis`. Run it on all
development seeds (`SWEEP="$TASK_DIR/debug/initial"`, tag `v00`). Preserve every executed
version; fixes get a new numbered file.

---

## Stage 1: Debug on the Development Seeds

You are improving ONE program. Debug by **failure mode, not by seed**. Every version runs on
**all three** development seeds. The number you optimize is how many of them pass.

### Step 1 — Group the failures

`$SWEEP` is the sweep's `--output-root`:

```bash
python3 -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + '/results.jsonl')]
for r in sorted(rows, key=lambda r: r['seed']):
    if not r['run_dir']:
        print(r['seed'], 'NO ARTIFACT', r['error']); continue
    o = json.load(open(r['run_dir'] + '/outcome.json'))
    print(r['seed'], o['program_ok'], o['task_success'], o['termination_reason'])
print('passed', sum(1 for r in rows if r['run_dir'] and
      json.load(open(r['run_dir'] + '/outcome.json'))['task_success']), 'of', len(rows))
" "$SWEEP"
```

A seed passes on `task_success` alone. Then read failed seeds in this order
(`docs/run-artifacts.md`):

- `outcome.json` — `program_ok`, `task_success`, `termination_reason`, `steps_executed`, `error`
- `source/program-result.json` — which public calls ran
- `trace/events.jsonl` — spans (`segment_text` with zero masks, navigation `ok=False`, joint
  error)
- `media/keyframes/`, `media/overlays/`, `media/videos/`, `episode/steps.jsonl`

| Mode | Evidence | Typical fix |
|---|---|---|
| Program | `program_error`, rejected AST, sandbox import | Smallest source defect. |
| Perception | Wrong/missing mask | Prompt fallback, other camera, check overlays. |
| Reach / base | Target outside the arm, forward nav `ok=False` | Lateral/yaw `navigate_to_pose`, then arm. |
| Execution | Trajectory or joints miss tolerance | Denser waypoints, `move_to_joints`, hold the goal. |
| Contact | Gripper on air, drawer unmoved | Staging, approach direction, close only at contact. |
| Task predicate | Program completes, `task_success=false` | Final geometry and incomplete subgoals. |

Write the grouping into `CHANGELOG.md`, most seeds first.

### Step 2 — Fix the largest mode, then run all development seeds

Write `$TASK_DIR/code/v<NN>_<slug>.py`. Never edit a version in place. Sweep with
`SWEEP="$TASK_DIR/debug/attempts/v<NN>"`. Record the pass count in CHANGELOG.

- **More passes than the best so far** — keep it and move to the next mode.
- **Same or fewer** — say which seeds regressed; repair or revert before continuing.

**Budget: 2 versions per mode, 4 in total** (v00 counts). A mode still unfixed after 2 is
blocked; write `$TASK_DIR/debug/blocked/<mode>.md` (`## Root Cause`, `## Details`,
`## What Was Tried`) and move on.

Probes: observation-only programs into `result`, `--output-root "$TASK_DIR/debug/explore"`
`--flat-run-dir --no-videos --init-mode saved`. Probes are not numbered versions.

### Step 3 — Select

Take the version with the most development passes; break ties by fewer `program_ok: false`,
then by the simpler program. Do not assemble a new program from pieces that never ran together.

Run the non-privileged provenance audit before selecting. If every version crashes, write a
minimal legal program (`get_observation()` and `result = {}`) so Stage 2 can still run.

Copy the selection to `$TASK_DIR/fix_code.py` and record it:
`SELECTED: v0N — n/3 development seeds`.

### Step 4 — Check it on the seed you have never run

Once, on 126:

```bash
"$CAP_HARNESS" run-batch \
  --benchmark robocasa --suite OpenDrawer --task-id 0 --seeds $UNSEEN_SEEDS \
  --program "$TASK_DIR/fix_code.py" \
  --output-root "$TASK_DIR/debug/holdout" --flat-run-dir --init-mode saved \
  --workers 1 --recycle-after 0 \
  --log-dir "$TASK_DIR/debug/logs/holdout" \
  --results-jsonl "$TASK_DIR/debug/holdout/results.jsonl"
```

Do not fix anything against 126 and do not run it twice.

### Step 5 — Write findings.md (REQUIRED)

```
# Findings: robocasa/OpenDrawer/task_0

## Root causes observed
## What fixed them
## Development vs unseen
## Generalizable patterns
## Blocked modes
## Non-privileged API audit
PASS, or QUARANTINED with the forbidden dependency
```

There is no skill library and no `skill_report.json` in this campaign.

---

## What to Return

You are done after Step 5. Do NOT run held-out seeds 1–3. Return exactly this summary:

```
BENCHMARK: robocasa
SUITE: OpenDrawer
TASK_ID: 0
GPU: <N>
SIM_PID: <contents of simulator.pid>
ENDPOINT: <CAP_HARNESS_ROBOCASA_ENDPOINT>

Stage 1 Results:
  Development seeds:  <n>/3   (v00 was <n>/3)
  Unseen check:        <0 or 1>/1
  Failure modes fixed:  <mode: seeds it cost>
  Failure modes blocked: <mode + root cause>
  fix_code.py written: yes/no
  findings.md written: yes/no
  Non-privileged API audit: PASS/QUARANTINED (<reason if quarantined>)

Key findings:
  <1–3 bullets: which failure mode cost the most, and what fixed it>
  <if 126 fails while development passed, say so — that is an overfitted program>
```

<!-- ==================== TEMPLATE END ==================== -->

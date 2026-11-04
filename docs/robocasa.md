# RoboCasa

RoboCasa kitchen tasks on a PandaOmron mobile manipulator, driven **across a process boundary**.

Unlike every other environment in this repo, the simulator is not imported. It runs in its own
process with its own mujoco/robosuite/robocasa, and `cap_harness.robocasa` reaches it over a raw
`step(action)` / observation loop — the same shape a real robot driver presents. Swapping the
simulator for hardware is a driver change, not a rewrite. See
[api-reference.md](api-reference.md) for the tool surface and
[run-artifacts.md](run-artifacts.md) for what a run writes.

```
aspire/.venv-robocasa  ──ZMQ DEALER→ROUTER (msgpack, ipc:///tmp/capx-sim-0.sock)──┐
  cap_harness.robocasa                                                            ▼
  NO mujoco/robosuite/robocasa                      robocasa/capx_sim_server (its own venv)

aspire/.venv-sam3      ──HTTP :8114 /segment, /segment_point, /healthz────────────┘
```

## Setup

```bash
scripts/bootstrap_robocasa.sh          # creates .venv-robocasa
scripts/bootstrap_sam3.sh --profile rtx5090   # creates .venv-sam3; see host-setup.md
source .venv-robocasa/bin/activate
```

SAM3 weights are gated. Request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), then `huggingface-cli login`. The checkpoint downloads on first service start. See [host-setup.md](host-setup.md).

The environment is deliberately tiny — cap-harness's four core dependencies, the transport, the
shared `robot-driver` package, and `imageio`/`pillow` for artifacts. The bootstrap **asserts
that mujoco, robosuite and robocasa are NOT importable**: if they were, a generated program
could reach simulator state directly and results would stop transferring to hardware. The
boundary is only structural if something fails when it is breached.

`MUJOCO_GL=egl` is **not** needed here. This side has no renderer; the simulator process owns
the only GL context and sets EGL for itself. `doctor` exempts this runtime accordingly.

## Running

SAM3 is a separate process on `http://127.0.0.1:8114`. The simulator does not serve it. From the workspace root:

```bash
scripts/ensure_sam3.sh    # starts SAM3 if :8114 is empty; never kills it
```

Then the simulator, still from the workspace root:

```bash
MUJOCO_GL=egl robocasa/capx_sim_server/.venv/bin/python \
    robocasa/capx_sim_server/server.py --task OpenDrawer
```

Then, from `aspire/`:

```bash
cap-harness doctor --runtime robocasa --expected-environment-name .venv-robocasa \
  --reset --reset-embodiment robocasa --reset-suite OpenDrawer \
  --output validation-artifacts/robocasa/doctor.json

cap-harness run --benchmark robocasa --suite OpenDrawer --task-id 0 --seed 101 \
  --init-mode saved --program examples/robocasa/connectivity_check.py --output-root outputs
```

`examples/robocasa/connectivity_check.py` is an **integration gate**, not a policy. It
measures that the public path actually moves and sees things (SAM on both cameras, gripper
open/close, `move_to_joints`, `execute_trajectory`, `go_home`, lateral + yaw base motion) and
raises if a required check fails, so `program_ok=False`. `task_success` will still be false —
the script does not try to open the drawer. `solve_ik`, `generate_grasps`, and an intermediate
`set_gripper` are asserted *absent* (RoboCasa has no IK/grasp backend and the jaw only holds
0 or 1). A green `program_ok` with `task_success=false` is the healthy result.

`--init-mode` must be `saved`; `seeded` is a LIBERO-only concept (it indexes LIBERO's saved
initial-state files). All RoboCasa variation comes from `--seed`.

**One simulator serves a whole campaign.** The adapter's `close()` disconnects without sending
the protocol's `CLOSE`, because `cap-harness run` closes the adapter after *every* episode and
`CLOSE` terminates the simulator process — episode 2 of a sweep would find nothing listening.
Tearing the simulator down is the supervisor's job.

## Registry

`OpenDrawer` and `PickPlaceCounterToStove`. `task_name` equals the environment name, so the same
string is the CLI's `--suite`, the registry key and the simulator's `--task`. `horizon` is
RoboCasa's own 1000 control ticks at 20 Hz; the simulator process is built with a larger
`max_steps` so one server can serve harnesses with different budgets.

## Seeds and scenes

A seed selects the kitchen. The partition is ASPIRE's Robosuite one — development **101-125**,
unseen-check **126-130**, held-out **1-100** — and the scene groups are RoboCasa's own:

| seeds | group | layouts | styles |
|---|---|---|---|
| >= 101 (dev, unseen-check) | `TRAIN` | 11-60 | 11-60 |
| 1-100 (held-out) | `TEST` | 1-10 | 1-10 |

The groups are disjoint, so a program debugged on development scenes is evaluated on kitchens it
has never been shown. That is a **stronger** split than the tabletop benchmarks use, where
development and held-out seeds differ only in value. The consequence is accepted knowingly:
ASPIRE's own Robosuite success rates are not a reference point for numbers produced this way.

`scene_group_for_seed()` owns that policy harness-side; the simulator only offers the
capability, so a caller with no partition protocol keeps the scene it started with.

Scene identity (`layout_id`, `style_id`) rides on the host-only `evaluate` channel and is kept
out of `TaskContext`, which generated programs can read.

## Cameras

`agentview` (the simulator maps it to `robot0_agentview_left`) and `robot0_eye_in_hand`. The
mapping is read from the simulator's `get_spec`, never hardcoded.

## Base motion

PandaOmron is mobile and the shared tool set has no base motion, so `robocasa.get_base_pose` and
`robocasa.navigate_to_pose` are public extensions. Without them every target beyond the arm's
~0.85 m reach from the spawn pose is unreachable.

BEHAVIOR's three standoff-geometry tools (`plan_approach_pose`, `base_pose_is_free`,
`plan_standoff_pose`) are deliberately **absent**: they need collision queries this simulator
does not serve. They are legitimate by ASPIRE's own precedent — a real mobile robot has a
costmap — so they are a missing service rather than forbidden knowledge.

## Things that will cost you time

1. **`robot0_base` is a decoy.** PandaOmron's nominal root is a dummy parked at world
   `[10, 10, 0]`; the real arm base is `robot0_link0`, which is what `BASE_FRAME` reports.
   Using the decoy puts every pose ~14 m away, silently.
2. **Never hardcode the action layout.** RoboCasa orders the composite controller's parts
   `right, right_gripper, base, torso`; robosuite's own `Lift` orders them
   `right, torso, base, right_gripper`. It is read from `get_spec` and assembled by the shared
   driver.
3. **The arm takes absolute joint targets**, not velocities: `pandaomron_joint_ctrl.json`
   configures `JOINT_POSITION` with `"input_type": "absolute"`. The Robosuite codec's
   `(target - current) * control_frequency` form would make the arm drift instead of track.
   The base, by contrast, is `JOINT_VELOCITY`, which is why base motion is a closed loop.
4. **Base stiction.** The base joints carry `frictionloss=250` against an actuator gain of
   1000, so commands below ~0.25 move nothing at all. The driver's `MotionProfile` floors each
   axis at 0.45; a plain proportional law stalls and never starts short moves.
5. **SAM3 must already be running.** `cap-harness run` checks `GET /healthz` on
   `:8114` before the program starts and exits with `SAM3 is not running on :8114`
   if it is not. That is infrastructure, not a policy failure. Start it with
   `scripts/ensure_sam3.sh` (workspace root) or
   `python -m cap_harness.providers.sam3.service`. `cabinet` and `counter` are
   reliable scene prompts; do not assume every English noun in the instruction
   will return a mask.
6. **`ndarray` methods break the program sandbox.** `points.mean(axis=0)`, `.min(axis=0)` and
   `.sum(axis=0)` trigger a lazy numpy import, and the sandbox passes `__builtins__` as a
   mappingproxy, so the import machinery fails with
   `'mappingproxy' object has no attribute '__import__'`. Module-level functions
   (`np.mean`, `np.median`, `np.linalg.norm`) and pure-Python idioms are fine. This is not
   specific to RoboCasa.
7. **`mask_to_point_cloud` needs an explicit `target_frame`.** Its default is the literal
   string `"robot_base"`, which matches no adapter's base frame. `localize_object` resolves the
   frame from the robot state and is the better entry point.
8. **`joint_velocities` are zeros.** The wire protocol carries no velocity channel. Nothing in
   the RoboCasa tool surface consumes them; if a planner ever does, the fix is a velocity field
   on the wire, not a finite difference in the adapter.

# RoboCasa CaP API tool check

Exercises **every one of the 31 public tools** the ASPIRE agent is handed for the `robocasa`
embodiment, and renders each one's input and output in a form a human can judge: masks drawn
on the picture they came from, point clouds and grasp poses in rotating 3D, and the simulator's
own video of every motion.

This is a *measurement*, not a policy and not a fix. Where a tool misbehaves it is reported
with the evidence and left alone.

## Where the tool list comes from

`cap_harness.registry.ToolRegistry` holds the allowlists --
`SHARED_PUBLIC_TOOL_NAMES` (24 names) and `ROBOCASA_PUBLIC_TOOL_NAMES` (7 `robocasa.*`
extensions) -- but it does not hold the implementations. The namespace a program actually
receives is assembled at runtime by `CapApi.register_tools` (`src/cap_harness/api.py`), which
binds 24 of its own methods plus every adapter method whose name is in its extension list,
prefixed with the adapter's embodiment. The bodies live in:

| tools | file |
| --- | --- |
| the 24 shared tools | `src/cap_harness/api.py` |
| the 7 `robocasa.*` extensions | `src/cap_harness/robocasa/adapter.py` |
| projection, cropping, OBB, top-down grasp selection | `src/cap_harness/geometry.py` |
| standoff / approach hull geometry | `src/cap_harness/robocasa/geometry.py` |
| SAM3, Contact-GraspNet, PyRoki, cuRobo wire clients | `src/cap_harness/providers/*/client.py` |

`probe.py` asserts that the assembled namespace is exactly 31 names rather than trusting the
constant.

## Files

| file | what it is |
| --- | --- |
| `session.py` | builds the same adapter + providers + `CapApi` + `ToolRegistry` that `cap_harness.run.run_program` builds for `robocasa`, plus a per-tick observer and dump helpers |
| `probe.py` | the measurements: nine stages, each with its own `reset()`, writing `out/raw/<stage>/` |
| `vis.py` | drawing primitives (projection, gripper and OBB wireframes, turntable GIFs) |
| `render.py` | turns `out/raw/` into `out/figures/`, `out/index.html` and `out/RESULTS.md` |
| `api_tool_check.py` | the same catalog written as a *generated program*, run inside `cap-harness` as a cross-check that the sandboxed path works too |
| `out/index.html` | **the report** -- open this |
| `out/SUMMARY.html` | the findings, pasted at the top of the report by `render.py` |
| `out/RESULTS.md` | one row per tool: checks made, checks that behaved |
| `out/raw/<stage>/` | records.json, arrays.npz, PNGs and the simulator's MP4s -- everything the figures are drawn from |
| `out/sandbox/` | the `cap-harness run` of `api_tool_check.py` (`program_ok=True`, 269 ticks) |
| `out/probe*.log` | probe stdout. `probe.log` is the full nine-stage run; `probe2.log` and `probe3.log` re-ran `identity`, `planning` and `graspexec` after those stages gained the joint-limit, stale-plan and pick-outcome checks, and are the source of those three stages' current data |

## Why the probe runs outside the program sandbox

`api_tool_check.py` runs where the agent's programs run: `runtime.py`'s sandbox, which injects
no builtins beyond `Exception`/`ValueError`/`TypeError`, permits `import math` and nothing
else, cannot write files, and breaks on ndarray *methods* (`points.mean(axis=0)`) because the
lazy numpy import fails against a `mappingproxy` `__builtins__`. That is the right place to
prove the agent-visible path works, and it is run for exactly that -- but it can produce only
text.

`probe.py` therefore calls **the same callables**, obtained the same way:

```python
registry = api.register_tools(ToolRegistry(public_extension_allowlist=ROBOCASA_PUBLIC_TOOL_NAMES))
tools = registry.public_tools()          # the mapping the sandbox is given
tools["segment_text"]("agentview", "fish")
```

Same adapter, same providers, same allowlist. The only difference is that this caller may keep
the arrays and draw them. Two places reach past that mapping, both marked in the source: the
simulator's video buffer (the public surface has no equivalent, and `navigate_to_pose` drives
inside the driver's own loop where the per-tick observer never fires) and `_elapsed_steps` for
the tick accounting in the log.

## Running it

Needs the simulator plus all four providers. From the workspace root:

```bash
./scripts/ensure_sam3.sh && ./scripts/ensure_pyroki.sh
./scripts/ensure_graspnet.sh && ./scripts/ensure_curobo.sh
MUJOCO_GL=egl robocasa/capx_sim_server/.venv/bin/python \
    robocasa/capx_sim_server/server.py --task PickPlaceCounterToStove --idle-timeout 0
```

Then, from `aspire/`:

```bash
# measure (~20 min for all nine stages; --stages <names> runs a subset)
PYTHONPATH="" .venv-robocasa/bin/python examples/robocasa/api_tool_check/probe.py \
    --task PickPlaceCounterToStove --seed 101

# draw. The harness venv deliberately has no plotting stack, so this one borrows the
# client venv, which already carries matplotlib. It only reads out/raw/.
/home/felix/robocasa_sim/cap-x/.venv-client/bin/python \
    examples/robocasa/api_tool_check/render.py

# cross-check through the real sandbox
PYTHONPATH="" .venv-robocasa/bin/cap-harness run --benchmark robocasa \
    --suite PickPlaceCounterToStove --task-id 0 --seed 101 --init-mode saved \
    --program examples/robocasa/api_tool_check/api_tool_check.py \
    --output-root examples/robocasa/api_tool_check/out/sandbox
```

`probe.py` resets the episode at the start of **every** stage, so each stage gets the full
1000-tick horizon and the identical kitchen (one seed), and a long motion stage cannot truncate
the one after it.

## Stages

| stage | tools it exercises |
| --- | --- |
| `identity` | `get_task_context`, `get_observation`, `get_robot_state`, `robocasa.get_task_metadata`, `robocasa.get_controller_metadata`, `robocasa.get_base_pose`, calibration round trip |
| `perception` | `segment_text`, `segment_points`, `mask_to_point_cloud`, `crop_point_cloud`, `estimate_geometry`, `localize_object` |
| `grasps` | `generate_grasps`, `select_grasp`, `solve_ik` on grasp poses |
| `graspexec` | drives the best candidate: `move_to_pose`, `close_gripper`, lift -- does the pipeline pick anything up? |
| `ik` | `solve_ik` identity, offset grid, reach sweep, and verification by driving each solution |
| `planning` | `plan_motion` across five strategies, `plan_synchronized_motion`, `execute_trajectory` |
| `gripper` | `open_gripper`, `close_gripper`, `set_gripper`, `set_grippers`, `step` (including refusals) |
| `motion` | `move_to_joints`, `execute_trajectory`, `move_to_pose` (four axes + rotation + unreachable), `move_synchronized`, `go_home` |
| `base` | `robocasa.base_pose_is_free` (961-call occupancy sweep), `robocasa.navigate_to_pose`, `robocasa.plan_standoff_pose`, `robocasa.plan_approach_pose` |

## Scope of this run

One task (`PickPlaceCounterToStove`) and one seed (101, a TRAIN-group kitchen). Switching
tasks means restarting `server.py` with a different `--task`, which was out of reach for this
run; `--task OpenDrawer --seed 101` is the same command once the simulator is serving it.
Nothing here is a claim about other kitchens.

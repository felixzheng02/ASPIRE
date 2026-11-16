# Integration gate for the OopsieVerse suites (Oopsie*). NOT a policy for the task.
#
# Runs the public API path a real program takes -- task context, observation, SAM on the
# scene camera, gripper, joint motion, go_home -- and measures that each step did something.
# It does not attempt the task, so task_success stays false. It never drives the base: spawn
# faces a counter and a blocked drive_base can run ~60 s.
#
# Health is scored by the harness outside the run directory; this program also checks that
# none of it is visible to a program.
#
# Sandbox: import math only. ndarray METHODS (points.mean(axis=0), .min, .sum) trigger a lazy
# numpy import and fail with "'mappingproxy' object has no attribute '__import__'".
import math

report = {}
failed = []

#: What SAM is asked to find for each gate suite; every suite also tries the counter.
TARGETS = {
    "OopsieOpenSingleDoor": "microwave",
    "OopsieTurnOnMicrowave": "microwave",
    "OopsieCounterToMicrowave": "cup",
    "OopsieTurnOnStove": "stove",
}
HIDDEN_TERMS = ("health", "safe", "damage")


def record(name, ok, detail):
    report[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        failed.append(name + ": " + str(detail))


def error_text(err):
    if err is None:
        return ""
    return str(err.message)


def joints_of(state):
    return [float(v) for v in state.joint_positions["primary"]]


def dist(a, b):
    total = 0.0
    for i in range(len(a)):
        d = float(a[i]) - float(b[i])
        total = total + d * d
    return math.sqrt(total)


def leaked_keys(mapping):
    found = []
    for key in mapping:
        lowered = str(key).lower()
        for term in HIDDEN_TERMS:
            if term in lowered:
                found.append(str(key))
    return found


# --------------------------------------------------------------------------- identity

context = get_task_context()
instruction = str(context.language)
record("instruction", len(instruction.strip()) > 0, instruction)
record("suite", str(context.suite).startswith("Oopsie"), str(context.suite))

metadata = robocasa.get_task_metadata()
leaks = leaked_keys(context.metadata) + leaked_keys(metadata)
record("no_health_visible", len(leaks) == 0, leaks or "none")

observation = get_observation()
record(
    "cameras",
    "agentview" in observation.cameras and "robot0_eye_in_hand" in observation.cameras,
    sorted(observation.cameras),
)

state = get_robot_state()
home_joints = joints_of(state)
record("base_frame", state.base_frame == "robot0_link0", state.base_frame)
record("n_joints", len(home_joints) == 7, len(home_joints))

# --------------------------------------------------------------------------- scene SAM

seen_any = False
for noun in [TARGETS.get(str(context.suite), "counter"), "counter"]:
    found = localize_object(noun, camera_name="agentview")
    n_points = int(len(found.point_cloud.points)) if found.ok else 0
    if found.ok and n_points >= 20 and found.point_cloud.frame == state.base_frame:
        seen_any = True
    report["localize_" + noun] = {
        "ok": bool(found.ok),
        "detail": {"n_points": n_points, "error": error_text(found.error)},
    }
record("localize_scene", seen_any, "a task object or the counter must localize")

# --------------------------------------------------------------------------- gripper

closed = close_gripper()
grip_closed = float(get_robot_state().gripper_positions["primary"])
record("close_gripper", closed.ok and grip_closed < 0.2,
       {"measured": round(grip_closed, 3), "error": error_text(closed.error)})

opened = open_gripper()
grip_open = float(get_robot_state().gripper_positions["primary"])
record("open_gripper", opened.ok and grip_open > 0.8,
       {"measured": round(grip_open, 3), "error": error_text(opened.error)})

# --------------------------------------------------------------------------- arm

before = joints_of(get_robot_state())
target = list(before)
target[1] = target[1] + 0.15
moved = move_to_joints(target, tolerance=0.02, max_steps=120)
after = joints_of(get_robot_state())
record("move_to_joints", moved.ok and abs(after[1] - target[1]) <= 0.05,
       {"joint1_after": round(after[1], 3), "joint1_target": round(target[1], 3),
        "error": error_text(moved.error)})

homed = go_home()
home_after = joints_of(get_robot_state())
record("go_home", homed.ok and dist(home_after, home_joints) <= 0.08,
       {"joint_error": round(dist(home_after, home_joints), 3), "error": error_text(homed.error)})

# --------------------------------------------------------------------------- gate

for key in sorted(report):
    print(key + ": " + str(report[key]))

result = report
if failed:
    raise Exception("oopsie gate failed: " + "; ".join(failed))

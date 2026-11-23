# Wrist sign check: from Home, pitch +0.3 rad and back, then roll +0.3 rad and back.
#
#   .venv-system2/bin/python -m cap_harness.stretch4.run_program --allow-motion \
#       examples/stretch4/wrist_joint_move.py
#
# Positive pitch tilts the wrist camera down. Positive roll turns the gripper
# clockwise when looking along the arm toward the fingertips. Watch the robot:
# each move should go that way, and each measured value should be within 3 deg
# of the command (the bridge fails a move that is not). Nothing else moves.
import math

STEP = 0.3  # rad, about 17 deg
PITCH, ROLL = 6, 7  # joint indices in the 8-joint Stretch arm

steps = []


def move_one_joint(label, index, value):
    joints = get_robot_state().joint_positions["primary"].copy()
    joints[index] = value
    outcome = move_to_joints(joints)
    measured = float(get_robot_state().joint_positions["primary"][index])
    steps.append(
        {"step": label, "ok": outcome.ok, "commanded": round(value, 4), "measured": round(measured, 4)}
    )
    print(
        f"{label}: commanded {value:+.4f} rad ({math.degrees(value):+.1f} deg), "
        f"measured {measured:+.4f} rad ({math.degrees(measured):+.1f} deg), ok {outcome.ok}"
    )
    if not outcome.ok:
        print(f"  error: {outcome.error}")
    return outcome.ok


home = go_home()
print(f"home: ok {home.ok}")
if home.ok:
    start = get_robot_state().joint_positions["primary"].copy()
    if move_one_joint("pitch +0.3 (camera down)", PITCH, start[PITCH] + STEP):
        move_one_joint("pitch back", PITCH, start[PITCH])
    if all(step["ok"] for step in steps) and move_one_joint(
        "roll +0.3 (clockwise toward fingertips)", ROLL, start[ROLL] + STEP
    ):
        move_one_joint("roll back", ROLL, start[ROLL])

result = {"home": home.ok, "steps": steps}

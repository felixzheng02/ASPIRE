# OopsieTurnOnMicrowave: press the microwave's start button.
#
# The scene camera sees the microwave but not its control panel (the arm hides its right side),
# so the wrist camera does the fine work: look at the right part of the front, find the panel,
# look again from closer with the image upright, collect button-like masks on the front plane,
# press the right-most button of the lowest row, retract.
#
# Text scores do not pick the start button reliably: SAM ranked an icon button (seed 101) and a
# preset button (seed 102) above it for "start". On both development panels seen, Start is the
# right-hand button of the bottom row, next to Stop; that layout prior is what this program uses.
#
# Sandbox: import math only; no ndarray methods.
import math

FORWARD = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]   # gripper approach axis along +x
UPRIGHT = [0.5, 0.5, 0.5, 0.5]                          # same axis, wrist image upright
PRESS_DEPTH_M = 0.015
ROW_TOLERANCE_M = 0.025


def centroid_of(points):
    n = len(points)
    return [sum([float(p[i]) for p in points]) / n for i in range(3)]


def quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))]


def front_masks(prompt, front_x, z_lo, z_hi, frame, min_points=20):
    """Wrist-camera masks for `prompt` whose points lie on the microwave front plane."""
    out = []
    seen = segment_text("robot0_eye_in_hand", prompt)
    for seg in seen.segmentations:
        cloud = mask_to_point_cloud(seg, "robot0_eye_in_hand", frame)
        if len(cloud.points) < min_points:
            continue
        c = centroid_of(cloud.points)
        if abs(c[0] - front_x) < 0.06 and z_lo - 0.08 <= c[2] <= z_hi + 0.08:
            out.append((float(seg.score), c))
    out.sort(key=lambda item: -item[0])
    return out


frame = get_robot_state().base_frame
found = localize_object("microwave", camera_name="agentview")
if not found.ok:
    raise Exception("microwave not localized")
pts = [[float(p[0]), float(p[1]), float(p[2])] for p in found.point_cloud.points]
z_lo = quantile([p[2] for p in pts], 0.10)
z_hi = quantile([p[2] for p in pts], 0.90)
body = [p for p in pts if z_lo <= p[2] <= z_hi]
front_x = quantile([p[0] for p in body], 0.05)
y_lo = quantile([p[1] for p in body], 0.05)
z_mid = 0.5 * (z_lo + z_hi)
print("microwave front_x", round(front_x, 3), "y_lo", round(y_lo, 3), "z", round(z_lo, 3), round(z_hi, 3))

close_gripper()

# 1. Look at the right part of the front; button masks there place the panel.
move_to_pose(Pose([front_x - 0.28, y_lo - 0.03, z_mid], FORWARD, frame), tolerance=0.02, max_steps=200)
buttons = front_masks("button", front_x, z_lo, z_hi, frame)
print("buttons on front:", [[round(s, 3), [round(v, 3) for v in c]] for s, c in buttons])
if buttons:
    panel_y = quantile([c[1] for s, c in buttons], 0.5)
    panel_z = quantile([c[2] for s, c in buttons], 0.5)
else:
    panel_y, panel_z = y_lo - 0.08, z_mid

# 2. Closer, upright look at the panel; the right-most button of the lowest row is the target.
move_to_pose(Pose([front_x - 0.20, panel_y, panel_z], UPRIGHT, frame), tolerance=0.02, max_steps=200)
candidates = front_masks("start", front_x, z_lo, z_hi, frame) + front_masks("button", front_x, z_lo, z_hi, frame)
print("button candidates:", [[round(s, 3), [round(v, 3) for v in c]] for s, c in candidates])
if not candidates:
    raise Exception("no button found on the microwave front")
lowest = min([c[2] for s, c in candidates])
bottom_row = [c for s, c in candidates if c[2] <= lowest + ROW_TOLERANCE_M]
target = min(bottom_row, key=lambda c: c[1])   # robot0_link0 y points left: min y is right-most
print("bottom row:", [[round(v, 3) for v in c] for c in bottom_row])
print("pressing", [round(v, 3) for v in target])

# 3. Approach square to the front, press a little past it, retract, go home.
move_to_pose(Pose([front_x - 0.10, target[1], target[2]], FORWARD, frame), tolerance=0.01, max_steps=200)
press = move_to_pose(Pose([front_x + PRESS_DEPTH_M, target[1], target[2]], FORWARD, frame),
                     tolerance=0.01, max_steps=40)
eef = get_robot_state().end_effector_poses["primary"]
print("press ok", press.ok, "eef", [round(float(v), 3) for v in eef.position])
move_to_pose(Pose([front_x - 0.15, target[1], target[2]], FORWARD, frame), tolerance=0.02, max_steps=200)
go_home()

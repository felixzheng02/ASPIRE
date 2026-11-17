# OopsieTurnOnMicrowave: press the microwave's start button.
#
# Approach
#   A. Scene camera: localize the microwave; take its front plane, right edge and height range.
#   B. Wrist camera, far look at the lower right of the front: segment the control panel as a
#      whole ("control panel" clears the 0.5 SAM3 score floor on every panel seen; "start" and
#      "button" often do not from this distance).
#   C. Close look at the bottom of the panel: measure the panel again and collect every small
#      mask on it for several element prompts (button, icon, word, number, ...). SAM returns at
#      most five masks per prompt, so several prompts are needed to cover the bottom rows.
#   D. One or two closer looks with the current pick near the top of the image, so that
#      anything lower on the panel is in view. Pick again from all views together.
#   E. Press the right-most element of the lowest row: approach square to the front, push a
#      little past the plane, retract, go home.
#
# What it assumes
#   - The microwave is in the scene camera's view at reset, its front faces the robot (the
#     front is a plane of constant robot0_link0 x), and the arm reaches the panel without
#     moving the base.
#   - The control panel is on the right side of the front.
#   - Start is the right-most element of the panel's lowest row. This is a layout prior, not
#     reading: it holds on the three development panels (icons 101, keypad 102, text labels
#     103). Text scores do not single out Start: "start" also returns the Stop button or
#     label, and on the icon panel it stays below the score floor.
#   - Rows of panel elements are more than ROW_TOLERANCE_M apart in height (the tightest
#     seen is 0.023 m, digits above the Stop / 0 / Start row of the keypad).
#   - The closed gripper hides the bottom sixth of the wrist image, and in the UPRIGHT pose
#     the wrist camera looks along the gripper axis from CAM_UP_M above it. Looks therefore
#     aim low, and each target is centred with that offset.
#
# Sandbox: import math only; no ndarray methods; no repr().
import math

FORWARD = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]   # gripper approach axis along +x
UPRIGHT = [0.5, 0.5, 0.5, 0.5]                          # same axis, wrist image upright
WRIST = "robot0_eye_in_hand"
SCENE = "agentview"
CAM_UP_M = 0.05          # in the UPRIGHT pose the wrist camera sits this far above the gripper axis
MIN_REACH_X = 0.15       # closer to the arm base than this, the wrist folds onto the robot
PRESS_DEPTH_M = 0.015
ROW_TOLERANCE_M = 0.02
PANEL_PROMPTS = ["microwave control panel", "control panel"]
ELEMENT_PROMPTS = ["button", "icon", "start button", "start", "stop button", "word", "number"]


def r3(values):
    return [round(float(v), 3) for v in values]


def quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))]


def cloud_stats(points):
    """5 / 50 / 95 % quantiles per axis of a (subsampled) cloud."""
    n = len(points)
    stride = max(1, n // 3000)
    xs = []
    ys = []
    zs = []
    for i in range(0, n, stride):
        p = points[i]
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
    return {"n": n,
            "x": [quantile(xs, 0.05), quantile(xs, 0.5), quantile(xs, 0.95)],
            "y": [quantile(ys, 0.05), quantile(ys, 0.5), quantile(ys, 0.95)],
            "z": [quantile(zs, 0.05), quantile(zs, 0.5), quantile(zs, 0.95)]}


def fmt(stats):
    return "n=" + str(stats["n"]) + " x=" + str(r3(stats["x"])) + " y=" + str(r3(stats["y"])) + " z=" + str(r3(stats["z"]))


def masks(camera, prompt, frame, tag):
    """[(score, stats)] for every mask of `prompt` that clears the harness score floor."""
    seen = segment_text(camera, prompt)
    if not seen.ok:
        print(tag, "|" + prompt + "|", "rejected:", seen.error.message[-36:])
        return []
    out = []
    for seg in seen.segmentations:
        try:
            stats = cloud_stats(mask_to_point_cloud(seg, camera, frame).points)
        except Exception as exc:
            continue
        print(tag, "|" + prompt + "|", round(float(seg.score), 3), "box", [int(float(v)) for v in seg.box_xyxy], fmt(stats))
        out.append((float(seg.score), stats))
    return out


def steps_left():
    return get_task_context().metadata["episode_steps_remaining"]


def reachable(position, quat, frame):
    return solve_ik(Pose(position, quat, frame)).ok


def move(tag, position, quat, frame, tolerance=0.02, max_steps=200):
    res = move_to_pose(Pose(position, quat, frame), tolerance=tolerance, max_steps=max_steps)
    eef = get_robot_state().end_effector_poses["primary"]
    print(tag, "target", r3(position), "ok", res.ok, "steps", res.steps_executed,
          "eef", r3(eef.position), "left", steps_left(),
          "" if res.error is None else res.error.message)
    return res


def look_at(tag, point, standoffs, frame):
    """Put the wrist camera axis on `point`, from the first standoff the arm can reach.

    If no standoff works, aim a little higher: low, close poses are the hard ones for the arm.
    """
    for raise_m in [0.0, 0.05, 0.10]:
        for standoff in standoffs:
            position = [point[0] - standoff, point[1], point[2] + raise_m - CAM_UP_M]
            if position[0] < MIN_REACH_X:
                continue
            if not reachable(position, UPRIGHT, frame):
                print(tag, "standoff", standoff, "raise", raise_m, "unreachable")
                continue
            res = move(tag, position, UPRIGHT, frame)
            if res.ok or res.steps_executed > 0:
                return True
    return False


def find_panel(tag, plane_x, near_y, frame):
    """Best control-panel mask on the microwave front, as bounds; None if no prompt clears."""
    best = None
    for prompt in PANEL_PROMPTS:
        for score, stats in masks(WRIST, prompt, frame, tag):
            width = stats["y"][2] - stats["y"][0]
            if abs(stats["x"][1] - plane_x) > 0.05 or not (0.04 <= width <= 0.25):
                continue
            if near_y is not None and not (stats["y"][0] - 0.03 <= near_y <= stats["y"][2] + 0.03):
                continue
            if best is None or score > best[0]:
                best = (score, stats)
    if best is None:
        return None
    stats = best[1]
    return {"x": stats["x"][1], "y_lo": stats["y"][0], "y_hi": stats["y"][2],
            "z_lo": stats["z"][0], "z_hi": stats["z"][2]}


def on_panel(stats, panel):
    """A small mask lying on the panel."""
    if stats["n"] < 15 or abs(stats["x"][1] - panel["x"]) > 0.02:
        return False
    if stats["y"][2] - stats["y"][0] > 0.05 or stats["z"][2] - stats["z"][0] > 0.05:
        return False
    if not (panel["y_lo"] - 0.01 <= stats["y"][1] <= panel["y_hi"] + 0.01):
        return False
    return panel["z_lo"] - 0.01 <= stats["z"][1] <= panel["z_hi"] + 0.02


def panel_elements(tag, panel, frame):
    """Small masks on the panel from the current wrist view: [(prompt, score, [x, y, z])]."""
    out = []
    for prompt in ELEMENT_PROMPTS:
        for score, stats in masks(WRIST, prompt, frame, tag):
            if on_panel(stats, panel):
                out.append((prompt, score, [stats["x"][1], stats["y"][1], stats["z"][1]]))
    return out


def lowest_right(elements):
    """Right-most element of the lowest row (robot0_link0 y points left)."""
    if not elements:
        return None
    lowest = min([e[2][2] for e in elements])
    row = [e for e in elements if e[2][2] <= lowest + ROW_TOLERANCE_M]
    return min(row, key=lambda e: e[2][1])


def show(tag, elements):
    print(tag, [[e[0], round(e[1], 2), r3(e[2])] for e in elements])


frame = get_robot_state().base_frame
print("language", get_task_context().language)

# A. Coarse: the microwave from the scene camera.
found = localize_object("microwave", camera_name=SCENE)
if not found.ok:
    raise Exception("microwave not localized")
pts = found.point_cloud.points
stride = max(1, len(pts) // 5000)
sample = [[float(pts[i][0]), float(pts[i][1]), float(pts[i][2])] for i in range(0, len(pts), stride)]
z_lo = quantile([p[2] for p in sample], 0.10)
z_hi = quantile([p[2] for p in sample], 0.90)
body = [p for p in sample if z_lo <= p[2] <= z_hi]
front_x = quantile([p[0] for p in body], 0.05)
y_lo = quantile([p[1] for p in body], 0.05)
print("microwave front_x", round(front_x, 3), "y_lo", round(y_lo, 3), "z", round(z_lo, 3), round(z_hi, 3))

# The panel as the scene camera sees it, when it does (the arm often hides it).
aim_y = y_lo - 0.03
for prompt in PANEL_PROMPTS:
    for score, stats in masks(SCENE, prompt, frame, "scene"):
        if abs(stats["x"][1] - front_x) < 0.05 and z_lo - 0.1 <= stats["z"][1] <= z_hi + 0.1:
            aim_y = stats["y"][1]
print("aim_y", round(aim_y, 3))

close_gripper()

# B. Far look with the wrist camera at the lower right of the front: the panel as a whole.
if not look_at("far", [front_x, aim_y, z_lo + 0.08], [0.32, 0.26, 0.20, 0.15, 0.10], frame):
    raise Exception("no reachable far view")
panel = find_panel("far", front_x, None, frame)
if panel is None:
    print("control panel not found; assuming a strip around the aim point")
    panel = {"x": front_x, "y_lo": aim_y - 0.07, "y_hi": aim_y + 0.07, "z_lo": z_lo, "z_hi": z_hi}
print("panel far", {k: round(v, 3) for k, v in panel.items()})
panel_y = 0.5 * (panel["y_lo"] + panel["y_hi"])

# C. Close look at the bottom of the panel; measure the panel again, list its small elements.
if not look_at("low", [panel["x"], panel_y, panel["z_lo"] + 0.03], [0.16, 0.13, 0.10, 0.07], frame):
    print("no reachable close view; using the far view")
closer = find_panel("low", panel["x"], panel_y, frame)
if closer is not None:
    panel = {"x": closer["x"], "y_lo": closer["y_lo"], "y_hi": closer["y_hi"],
             "z_lo": min(panel["z_lo"], closer["z_lo"]), "z_hi": max(panel["z_hi"], closer["z_hi"])}
    print("panel low", {k: round(v, 3) for k, v in panel.items()})
elements = panel_elements("low", panel, frame)
show("elements low", elements)
pick = lowest_right(elements)
if pick is None:
    # Nothing found low on the panel: look higher.
    look_at("mid", [panel["x"], panel_y, panel["z_lo"] + 0.15], [0.16, 0.13, 0.10, 0.07], frame)
    elements = panel_elements("mid", panel, frame)
    show("elements mid", elements)
    pick = lowest_right(elements)
if pick is None:
    raise Exception("no button found on the control panel")
print("first pick", pick[0], r3(pick[2]))

# D. Closer looks with the pick near the top of the image, so that anything lower on the
#    panel is in view and the five masks SAM returns per prompt are spent on the bottom rows.
for attempt in range(2):
    if not look_at("near", [pick[2][0], pick[2][1], pick[2][2] - 0.04], [0.11, 0.09, 0.07, 0.14], frame):
        break
    near = panel_elements("near", panel, frame)
    show("elements near", near)
    elements = elements + near
    again = lowest_right(elements)
    moved = abs(again[2][1] - pick[2][1]) + abs(again[2][2] - pick[2][2])
    pick = again
    print("pick", attempt, pick[0], r3(pick[2]), "moved", round(moved, 3))
    if moved < 0.015:
        break
target = pick[2]
print("pressing", r3(target))

# E. Approach square to the front, press a little past it, retract, go home.
quat = FORWARD
if not reachable([panel["x"] + PRESS_DEPTH_M, target[1], target[2]], FORWARD, frame):
    print("FORWARD press unreachable; using UPRIGHT")
    quat = UPRIGHT
move("approach", [panel["x"] - 0.10, target[1], target[2]], quat, frame, tolerance=0.01)
press = move_to_pose(Pose([panel["x"] + PRESS_DEPTH_M, target[1], target[2]], quat, frame),
                     tolerance=0.01, max_steps=40)
eef = get_robot_state().end_effector_poses["primary"]
print("press ok", press.ok, "steps", press.steps_executed, "eef", r3(eef.position), "left", steps_left())
move("retract", [panel["x"] - 0.15, target[1], target[2]], quat, frame)
go_home()

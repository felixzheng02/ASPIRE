# Catalog of every public RoboCasa CaP tool. NOT a task policy.
#
# Prints a structured dump of each return value. Image-like results are also
# saved by cap-harness (the sandbox cannot write files):
#   logs/program.stdout           this dump, including ASCII maps
#   media/overlays/sam3-*.png     segment_text / segment_points
#   media/overlays/grasp-*.png    generate_grasps
#   media/keyframes/              RGB before/after every motion call
#   media/videos/                 episode cameras
#   source/program-result.json    the `result` mapping below
#
# Run from aspire/:
#   PYTHONPATH="" .venv-robocasa/bin/cap-harness run --benchmark robocasa \
#     --suite OpenDrawer --task-id 0 --seed 101 --init-mode saved \
#     --program examples/robocasa/api_tool_check.py --output-root outputs
#
# Sandbox: import math only. ndarray METHODS (points.mean(axis=0), .min, .sum)
# trigger a lazy numpy import and fail with
# `'mappingproxy' object has no attribute '__import__'`.
import math

report = {}
failed = []

EXPECTED = [
    "get_task_context",
    "get_observation",
    "get_robot_state",
    "segment_text",
    "segment_points",
    "mask_to_point_cloud",
    "crop_point_cloud",
    "estimate_geometry",
    "localize_object",
    "generate_grasps",
    "select_grasp",
    "solve_ik",
    "plan_motion",
    "plan_synchronized_motion",
    "step",
    "execute_trajectory",
    "set_gripper",
    "set_grippers",
    "move_to_joints",
    "move_to_pose",
    "move_synchronized",
    "open_gripper",
    "close_gripper",
    "go_home",
    "robocasa.get_task_metadata",
    "robocasa.get_controller_metadata",
    "robocasa.get_base_pose",
    "robocasa.navigate_to_pose",
    "robocasa.base_pose_is_free",
    "robocasa.plan_standoff_pose",
    "robocasa.plan_approach_pose",
]


def record(name, ok, detail):
    report[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        failed.append(name)


def error_text(err):
    if err is None:
        return ""
    return str(err.code) + ": " + str(err.message)


def rounds(values, digits=4):
    out = []
    for value in values:
        out.append(round(float(value), digits))
    return out


def joints_of(state):
    return [float(v) for v in state.joint_positions["primary"]]


def names_of(state):
    return [str(name) for name in state.joint_names["primary"]]


def dist(a, b):
    total = 0.0
    for i in range(len(a)):
        d = float(a[i]) - float(b[i])
        total = total + d * d
    return math.sqrt(total)


def yaw_of(pose):
    w, x, y, z = [float(v) for v in pose.quaternion_wxyz]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_pi(angle):
    while angle > math.pi:
        angle = angle - 2.0 * math.pi
    while angle < -math.pi:
        angle = angle + 2.0 * math.pi
    return angle


def centroid_of(points):
    n = len(points)
    return [sum([float(p[i]) for p in points]) / n for i in range(3)]


def aabb_of(points):
    mins = [float(points[0][0]), float(points[0][1]), float(points[0][2])]
    maxs = [mins[0], mins[1], mins[2]]
    for p in points:
        for i in range(3):
            v = float(p[i])
            if v < mins[i]:
                mins[i] = v
            if v > maxs[i]:
                maxs[i] = v
    return {"min": rounds(mins, 3), "max": rounds(maxs, 3)}


def dump_pose(pose):
    return {
        "position_m": rounds(pose.position),
        "quaternion_wxyz": rounds(pose.quaternion_wxyz),
        "yaw_rad": round(yaw_of(pose), 4),
        "frame": str(pose.frame),
    }


def dump_execution(result):
    errors = {}
    for key in result.final_errors:
        errors[str(key)] = round(float(result.final_errors[key]), 4)
    return {
        "ok": bool(result.ok),
        "status": str(result.status),
        "steps_executed": int(result.steps_executed),
        "terminated": bool(result.terminated),
        "truncated": bool(result.truncated),
        "final_errors": errors,
        "error": error_text(result.error),
    }


def dump_step(result):
    return {
        "ok": bool(result.ok),
        "terminated": bool(result.terminated),
        "truncated": bool(result.truncated),
        "error": error_text(result.error),
    }


def mapping_text(value):
    out = {}
    for key in value:
        out[str(key)] = str(value[key])
    return out


def mask_pixels(mask):
    total = 0
    for row in mask.tolist():
        for value in row:
            if value:
                total = total + 1
    return int(total)


def mask_ascii(mask, max_cols=48):
    rows = mask.tolist()
    height = len(rows)
    width = len(rows[0])
    step_x = max(1, width // max_cols)
    step_y = max(1, step_x * 2)
    lines = []
    y = 0
    while y < height:
        chars = []
        x = 0
        while x < width:
            chars.append("#" if rows[y][x] else ".")
            x = x + step_x
        lines.append("".join(chars))
        y = y + step_y
    return lines


def cloud_ascii(points, cols=41, rows=13):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    if xmax - xmin < 1e-3:
        xmax = xmin + 1e-3
    if ymax - ymin < 1e-3:
        ymax = ymin + 1e-3
    grid = []
    for row_i in range(rows):
        grid.append(["."] * cols)
    for i in range(len(xs)):
        c = int((xs[i] - xmin) / (xmax - xmin) * (cols - 1))
        r = int((ymax - ys[i]) / (ymax - ymin) * (rows - 1))
        if r < 0:
            r = 0
        if r >= rows:
            r = rows - 1
        if c < 0:
            c = 0
        if c >= cols:
            c = cols - 1
        grid[r][c] = "#"
    lines = [
        "x=["
        + str(round(xmin, 3))
        + ", "
        + str(round(xmax, 3))
        + "]  y=["
        + str(round(ymin, 3))
        + ", "
        + str(round(ymax, 3))
        + "]  (top = +y)"
    ]
    for row in grid:
        lines.append("".join(row))
    return lines


def dump_segmentation(seg):
    box = None
    if seg.box_xyxy is not None:
        box = rounds(seg.box_xyxy, 1)
    return {
        "label": str(seg.label),
        "score": round(float(seg.score), 3),
        "camera_name": str(seg.camera_name),
        "frame": str(seg.frame),
        "box_xyxy": box,
        "mask_hw": [int(seg.mask.shape[0]), int(seg.mask.shape[1])],
        "mask_pixels": mask_pixels(seg.mask),
    }


def dump_cloud(cloud):
    points = cloud.points
    return {
        "n_points": int(len(points)),
        "frame": str(cloud.frame),
        "centroid_m": rounds(centroid_of(points), 3),
        "aabb_m": aabb_of(points),
        "has_colors": cloud.colors is not None,
    }


def dump_geometry(geometry):
    return {
        "pose": dump_pose(geometry.pose),
        "extents_m": rounds(geometry.extents, 3),
        "point_count": geometry.point_count,
        "frame": str(geometry.frame),
    }


def dump_camera(name, camera):
    rgb = camera.rgb
    depth = camera.depth_m
    h = int(rgb.shape[0])
    w = int(rgb.shape[1])
    cy = h // 2
    cx = w // 2
    k = camera.intrinsics
    return {
        "name": name,
        "rgb_hwc": [h, w, int(rgb.shape[2])],
        "center_rgb": [
            int(round(float(rgb[cy, cx, 0]))),
            int(round(float(rgb[cy, cx, 1]))),
            int(round(float(rgb[cy, cx, 2]))),
        ],
        "center_depth_m": round(float(depth[cy, cx]), 3),
        "frame": str(camera.frame),
        "camera_pose": dump_pose(camera.camera_pose),
        "intrinsics": [
            rounds(k[0], 2),
            rounds(k[1], 2),
            rounds(k[2], 2),
        ],
    }


def dump_grasp(grasp):
    return {
        "score": round(float(grasp.score), 3),
        "width_m": round(float(grasp.width_m), 3),
        "approach_distance_m": round(float(grasp.approach_distance_m), 3),
        "pose": dump_pose(grasp.pose),
    }


def qrot(q, v):
    w, x, y, z = [float(a) for a in q]
    vx, vy, vz = [float(a) for a in v]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def print_lines(title, lines):
    print(title)
    for line in lines:
        print("  " + line)


def call(name, fn):
    print("")
    print("=" * 72)
    print(name)
    print("=" * 72)
    try:
        ok, detail = fn()
    except Exception as exc:
        ok = False
        detail = {"error": "raised: " + str(exc)}
        print("raised: " + str(exc))
    record(name, ok, detail)
    print("ok: " + str(ok))
    print(detail)
    return ok


# --------------------------------------------------------------------------- identity / metadata
# Nested functions cannot assign module names (sandbox bans `global`). Stash
# shared episode state on this dict instead.

hold = {}


def do_get_task_context():
    context = get_task_context()
    hold["context"] = context
    detail = {
        "suite": str(context.suite),
        "task_id": int(context.task_id),
        "task_name": str(context.task_name),
        "family": str(context.family),
        "language": str(context.language),
        "metadata": mapping_text(context.metadata),
    }
    return len(str(context.language).strip()) > 0, detail


def do_get_observation():
    observation = get_observation()
    hold["observation"] = observation
    cameras = []
    for name in sorted(observation.cameras):
        cameras.append(dump_camera(name, observation.cameras[name]))
    filtered = get_observation("agentview")
    detail = {
        "camera_names": sorted(observation.cameras),
        "cameras": cameras,
        "filtered_agentview_only": sorted(filtered.cameras),
        "timestamp_s": observation.timestamp_s,
    }
    return ("agentview" in observation.cameras) and (
        "robot0_eye_in_hand" in observation.cameras
    ), detail


def do_get_robot_state():
    state = get_robot_state()
    hold["state"] = state
    hold["home_joints"] = joints_of(state)
    hold["frame"] = str(state.base_frame)
    hold["ee"] = state.end_effector_poses["primary"]
    detail = {
        "embodiment": str(state.embodiment),
        "base_frame": hold["frame"],
        "arms": [str(a) for a in state.arms],
        "joint_names": names_of(state),
        "joint_positions_rad": rounds(hold["home_joints"], 4),
        "joint_velocities": rounds(state.joint_velocities["primary"], 4),
        "gripper_position": round(float(state.gripper_positions["primary"]), 3),
        "ee_pose": dump_pose(hold["ee"]),
        "timestamp_s": state.timestamp_s,
    }
    return hold["frame"] == "robot0_link0" and len(hold["home_joints"]) == 7, detail


def do_task_metadata():
    meta = robocasa.get_task_metadata()
    detail = mapping_text(meta)
    return "task_name" in meta, detail


def do_controller_metadata():
    meta = robocasa.get_controller_metadata()
    detail = mapping_text(meta)
    return bool(meta["mobile_base"]) and str(meta["base_frame"]) == "robot0_link0", detail


call("get_task_context", do_get_task_context)
call("get_observation", do_get_observation)
call("get_robot_state", do_get_robot_state)
call("robocasa.get_task_metadata", do_task_metadata)
call("robocasa.get_controller_metadata", do_controller_metadata)


# --------------------------------------------------------------------------- perception
# Reliable scene nouns on this kitchen: cabinet / counter -- on agentview. The wrist is
# prompted with the same two nouns on purpose, and both are expected to come back
# ok=False: segment_text enforces a 0.5 presence floor on robocasa, and from
# robot0_eye_in_hand the cabinet scores 0.049 and the counter 0.398 because neither is
# really in that frame. The check passes on the agentview hits; the wrist rejections are
# recorded with the score that caused them.

def best_candidate(found):
    """SAM3's top score before filtering, wherever this result carries it."""
    if found.ok:
        return found.diagnostics.get("best_candidate_score")
    if found.error is None:
        return None
    return found.error.details.get("best_candidate_score")


def do_segment_text():
    by_camera = {}
    ok = False
    for camera_name in ["agentview", "robot0_eye_in_hand"]:
        hits = []
        for prompt in ["cabinet", "counter"]:
            found = segment_text(camera_name, prompt)
            score = best_candidate(found)
            entry = {
                "prompt": prompt,
                "ok": bool(found.ok),
                "n": 0 if not found.ok else len(found.segmentations),
                "best_candidate_score": None if score is None else round(float(score), 4),
                "error": error_text(found.error),
                "top": None,
            }
            if found.ok and found.segmentations:
                top = found.segmentations[0]
                entry["top"] = dump_segmentation(top)
                print_lines(
                    camera_name + " / " + prompt + " mask (harness also writes sam3 PNG)",
                    mask_ascii(top.mask),
                )
                if camera_name == "agentview" and prompt == "cabinet" and hold.get("cabinet_mask") is None:
                    hold["cabinet_mask"] = top
                ok = True
            hits.append(entry)
        by_camera[camera_name] = hits
    return ok, by_camera


def do_segment_points():
    observation = hold.get("observation")
    if observation is None:
        return False, "no observation"
    wrist = observation.cameras["robot0_eye_in_hand"]
    h = int(wrist.rgb.shape[0])
    w = int(wrist.rgb.shape[1])
    clicked = segment_points("robot0_eye_in_hand", [[float(w) / 2.0, float(h) / 2.0]])
    detail = {
        "point_px": [round(float(w) / 2.0, 1), round(float(h) / 2.0, 1)],
        "ok": bool(clicked.ok),
        "n": 0 if not clicked.ok else len(clicked.segmentations),
        "error": error_text(clicked.error),
        "top": None,
    }
    if clicked.ok and clicked.segmentations:
        top = clicked.segmentations[0]
        detail["top"] = dump_segmentation(top)
        print_lines("wrist center-click mask", mask_ascii(top.mask))
        return True, detail
    return False, detail


def do_mask_to_point_cloud():
    observation = hold.get("observation")
    state = hold.get("state")
    if observation is None or state is None:
        return False, "no observation"
    found = segment_text("agentview", "counter")
    if not found.ok or not found.segmentations:
        return False, {"error": error_text(found.error) or "empty counter mask"}
    cloud = mask_to_point_cloud(found.segmentations[0], "agentview", state.base_frame)
    hold["counter_cloud"] = cloud
    detail = dump_cloud(cloud)
    print_lines("counter cloud XY (frame " + cloud.frame + ")", cloud_ascii(cloud.points))
    return int(len(cloud.points)) >= 20 and cloud.frame == state.base_frame, detail


def do_localize_object():
    state = hold.get("state")
    if state is None:
        return False, "no robot state"
    found = localize_object("cabinet", camera_name="agentview", target_frame=state.base_frame)
    detail = {"ok": bool(found.ok), "error": error_text(found.error)}
    if not found.ok:
        return False, detail
    hold["cabinet_cloud"] = found.point_cloud
    if found.segmentation is not None:
        hold["cabinet_mask"] = found.segmentation
        detail["segmentation"] = dump_segmentation(found.segmentation)
    detail["point_cloud"] = dump_cloud(found.point_cloud)
    detail["geometry"] = dump_geometry(found.geometry)
    print_lines(
        "cabinet cloud XY (localize_object, frame " + found.point_cloud.frame + ")",
        cloud_ascii(found.point_cloud.points),
    )
    return int(len(found.point_cloud.points)) >= 20, detail


def do_crop_point_cloud():
    cabinet_cloud = hold.get("cabinet_cloud")
    if cabinet_cloud is None:
        return False, "no cabinet cloud"
    c = centroid_of(cabinet_cloud.points)
    cropped = crop_point_cloud(
        cabinet_cloud,
        [c[0] - 2.0, c[1] - 2.0, c[2] - 2.0],
        [c[0] + 2.0, c[1] + 2.0, c[2] + 2.0],
    )
    detail = dump_cloud(cropped)
    return int(len(cropped.points)) >= 20, detail


def do_estimate_geometry():
    cabinet_cloud = hold.get("cabinet_cloud")
    if cabinet_cloud is None:
        return False, "no cabinet cloud"
    geometry = estimate_geometry(cabinet_cloud)
    detail = dump_geometry(geometry)
    return float(geometry.extents[0]) > 0.0, detail


call("segment_text", do_segment_text)
call("segment_points", do_segment_points)
call("mask_to_point_cloud", do_mask_to_point_cloud)
call("localize_object", do_localize_object)
call("crop_point_cloud", do_crop_point_cloud)
call("estimate_geometry", do_estimate_geometry)


# --------------------------------------------------------------------------- mobile base geometry

def do_get_base_pose():
    spawn = robocasa.get_base_pose()
    hold["spawn"] = spawn
    detail = dump_pose(spawn)
    return spawn.frame == "world", detail


def do_base_pose_is_free():
    spawn = hold.get("spawn")
    if spawn is None:
        return False, "no spawn pose"
    sx = float(spawn.position[0])
    sy = float(spawn.position[1])
    yaw = yaw_of(spawn)
    spawn_free = robocasa.base_pose_is_free(sx, sy, yaw)
    half = 0.75
    step = 0.25
    n = 7
    ascii_rows = []
    blocked = 0
    free = 0
    i = 0
    while i < n:
        y = sy + half - i * step
        chars = []
        j = 0
        while j < n:
            x = sx - half + j * step
            is_free = robocasa.base_pose_is_free(x, y, yaw)
            if i == n // 2 and j == n // 2:
                chars.append("R")
            elif is_free:
                chars.append(".")
                free = free + 1
            else:
                chars.append("#")
                blocked = blocked + 1
            j = j + 1
        ascii_rows.append("".join(chars))
        i = i + 1
    print_lines(
        "world occupancy around spawn (. free  # blocked  R robot)  "
        + str(step)
        + " m/cell, yaw="
        + str(round(yaw, 3)),
        ascii_rows,
    )
    counter_query = None
    counter_cloud = hold.get("counter_cloud")
    if counter_cloud is not None:
        c = centroid_of(counter_cloud.points)
        rotated = qrot(spawn.quaternion_wxyz, c)
        cx = sx + rotated[0]
        cy = sy + rotated[1]
        counter_free = robocasa.base_pose_is_free(cx, cy, yaw)
        counter_query = {
            "x": round(cx, 3),
            "y": round(cy, 3),
            "free": bool(counter_free),
        }
    detail = {
        "spawn": {"x": round(sx, 3), "y": round(sy, 3), "yaw": round(yaw, 3), "free": spawn_free},
        "grid": {
            "half_m": half,
            "step_m": step,
            "n": n,
            "free_cells": free,
            "blocked_cells": blocked,
            "ascii": ascii_rows,
        },
        "counter_centroid_world": counter_query,
    }
    return spawn_free is True, detail


def do_plan_standoff():
    counter_cloud = hold.get("counter_cloud")
    cabinet_cloud = hold.get("cabinet_cloud")
    if counter_cloud is None or cabinet_cloud is None:
        return False, "need counter and cabinet clouds"
    standoff = robocasa.plan_standoff_pose(counter_cloud, cabinet_cloud, 0.3)
    detail = {
        "x": round(float(standoff[0]), 3),
        "y": round(float(standoff[1]), 3),
        "yaw": round(float(standoff[2]), 3),
        "frame_note": "same as the clouds (" + str(counter_cloud.frame) + ")",
    }
    return all(math.isfinite(float(v)) for v in standoff), detail


def do_plan_approach():
    cabinet_cloud = hold.get("cabinet_cloud")
    if cabinet_cloud is None:
        return False, "need a cabinet cloud"
    approach = robocasa.plan_approach_pose(cabinet_cloud, 0.7)
    detail = {
        "x": round(float(approach[0]), 3),
        "y": round(float(approach[1]), 3),
        "yaw": round(float(approach[2]), 3),
        "frame_note": "same as the cloud (" + str(cabinet_cloud.frame) + ")",
    }
    return all(math.isfinite(float(v)) for v in approach), detail


call("robocasa.get_base_pose", do_get_base_pose)
call("robocasa.base_pose_is_free", do_base_pose_is_free)
call("robocasa.plan_standoff_pose", do_plan_standoff)
call("robocasa.plan_approach_pose", do_plan_approach)


# --------------------------------------------------------------------------- grasps (GraspNet :8115). Overlay PNG is written by the harness.

def do_generate_grasps():
    cabinet_mask = hold.get("cabinet_mask")
    if cabinet_mask is None:
        return False, "no cabinet mask"
    grasps = generate_grasps("agentview", cabinet_mask, max_candidates=5)
    hold["last_grasps"] = grasps
    n = 0 if (not grasps.ok or grasps.grasps is None) else len(grasps.grasps)
    candidates = []
    if n > 0:
        for grasp in grasps.grasps:
            candidates.append(dump_grasp(grasp))
    detail = {
        "ok": bool(grasps.ok),
        "n": n,
        "error": error_text(grasps.error),
        "candidates": candidates,
        "diagnostics": mapping_text(grasps.diagnostics),
    }
    return grasps.ok and n > 0, detail


def do_select_grasp():
    grasps = hold.get("last_grasps")
    cabinet_mask = hold.get("cabinet_mask")
    if grasps is None and cabinet_mask is not None:
        grasps = generate_grasps("agentview", cabinet_mask, max_candidates=5)
    if grasps is None:
        return False, "no grasps"
    if not grasps.ok:
        return False, {"error": error_text(grasps.error)}
    top_down = select_grasp(grasps, "top_down")
    highest = select_grasp(grasps, "highest_score")
    detail = {
        "top_down": None if top_down is None else dump_grasp(top_down),
        "highest_score": None if highest is None else dump_grasp(highest),
        "note": "top_down is often empty on cabinets; highest_score is the usual choice",
    }
    return highest is not None, detail


call("generate_grasps", do_generate_grasps)
call("select_grasp", do_select_grasp)


# --------------------------------------------------------------------------- kinematics / planning (no arm motion yet)

def do_solve_ik():
    current = get_robot_state()
    pose = current.end_effector_poses["primary"]
    ik = solve_ik(pose)
    detail = {
        "ok": bool(ik.ok),
        "arm": str(ik.arm),
        "embodiment": str(ik.embodiment),
        "n_joints": 0 if ik.joint_positions is None else len(ik.joint_positions),
        "joint_positions_rad": None
        if ik.joint_positions is None
        else rounds(ik.joint_positions, 4),
        "error": error_text(ik.error),
        "target": dump_pose(pose),
    }
    return ik.ok, detail


def do_plan_motion():
    current = joints_of(get_robot_state())
    goal = list(current)
    goal[2] = current[2] + 0.06
    interpolated = plan_motion(goal)
    n_interp = 0
    interp_path = None
    if interpolated.trajectory is not None:
        n_interp = len(interpolated.trajectory.joint_positions)
        interp_path = {
            "n_waypoints": n_interp,
            "collision_aware": bool(interpolated.trajectory.collision_aware),
            "planner": str(interpolated.trajectory.planner),
            "dt_s": float(interpolated.trajectory.dt_s),
            "start": rounds(interpolated.trajectory.joint_positions[0], 4),
            "end": rounds(interpolated.trajectory.joint_positions[n_interp - 1], 4),
        }
    curobo_plan = plan_motion(
        goal,
        strategy=MotionStrategy(ik_solver="pyroki", trajectory_planner="curobo"),
    )
    n_curobo = 0
    curobo_path = None
    if curobo_plan.trajectory is not None:
        n_curobo = len(curobo_plan.trajectory.joint_positions)
        curobo_path = {
            "n_waypoints": n_curobo,
            "collision_aware": bool(curobo_plan.trajectory.collision_aware),
            "planner": str(curobo_plan.trajectory.planner),
            "start": rounds(curobo_plan.trajectory.joint_positions[0], 4),
            "end": rounds(curobo_plan.trajectory.joint_positions[n_curobo - 1], 4),
        }
    detail = {
        "interpolation": {
            "ok": bool(interpolated.ok),
            "error": error_text(interpolated.error),
            "trajectory": interp_path,
        },
        "curobo": {
            "ok": bool(curobo_plan.ok),
            "error": error_text(curobo_plan.error),
            "trajectory": curobo_path,
        },
    }
    return interpolated.ok and n_interp >= 2, detail


def do_plan_synchronized():
    pose = get_robot_state().end_effector_poses["primary"]
    planned = plan_synchronized_motion({"primary": pose})
    n = 0
    path = None
    if planned.ok and planned.trajectory is not None:
        n = len(planned.trajectory.joint_positions)
        path = {
            "n_waypoints": n,
            "planner": str(planned.trajectory.planner),
            "collision_aware": bool(planned.trajectory.collision_aware),
        }
    detail = {
        "ok": bool(planned.ok),
        "error": error_text(planned.error),
        "trajectory": path,
        "note": "unimanual; a typed failure is still a successful probe of this tool",
    }
    return True, detail


call("solve_ik", do_solve_ik)
call("plan_motion", do_plan_motion)
call("plan_synchronized_motion", do_plan_synchronized)


# --------------------------------------------------------------------------- gripper + one-tick step

def do_close_gripper():
    closed = close_gripper()
    measured = float(get_robot_state().gripper_positions["primary"])
    detail = dump_execution(closed)
    detail["measured"] = round(measured, 3)
    return closed.ok and measured < 0.2, detail


def do_open_gripper():
    opened = open_gripper()
    measured = float(get_robot_state().gripper_positions["primary"])
    detail = dump_execution(opened)
    detail["measured"] = round(measured, 3)
    return opened.ok and measured > 0.8, detail


def do_set_gripper():
    opened = set_gripper(1.0)
    closed = set_gripper(0.0)
    mid = set_gripper(0.5)
    reopen = set_gripper(1.0)
    detail = {
        "open_1.0": dump_execution(opened),
        "close_0.0": dump_execution(closed),
        "mid_0.5": dump_execution(mid),
        "reopen_1.0": dump_execution(reopen),
        "note": "mid-range must be unsupported: this jaw only holds 0 or 1",
    }
    mid_ok = (not mid.ok) and ("unsupported" in error_text(mid.error).lower())
    return opened.ok and closed.ok and reopen.ok and mid_ok, detail


def do_set_grippers():
    result = set_grippers({"primary": 1.0})
    measured = float(get_robot_state().gripper_positions["primary"])
    detail = dump_execution(result)
    detail["measured"] = round(measured, 3)
    return result.ok, detail


def do_step():
    current = get_robot_state()
    action = RobotAction(
        arms={
            "primary": ArmCommand(
                mode="joint_position",
                target=joints_of(current),
                gripper_position=float(current.gripper_positions["primary"]),
                embodiment="robocasa",
            )
        }
    )
    result = step(action)
    after = joints_of(get_robot_state())
    detail = dump_step(result)
    detail["joint_drift_rad"] = round(dist(after, joints_of(current)), 4)
    return result.ok, detail


call("close_gripper", do_close_gripper)
call("open_gripper", do_open_gripper)
call("set_gripper", do_set_gripper)
call("set_grippers", do_set_grippers)
call("step", do_step)


# --------------------------------------------------------------------------- small arm motion (keyframes written around each call)

def do_move_to_joints():
    before = joints_of(get_robot_state())
    target = list(before)
    target[1] = target[1] + 0.12
    moved = move_to_joints(target, tolerance=0.02, max_steps=120)
    after = joints_of(get_robot_state())
    detail = dump_execution(moved)
    detail["joint1_before"] = round(before[1], 4)
    detail["joint1_target"] = round(target[1], 4)
    detail["joint1_after"] = round(after[1], 4)
    return moved.ok and abs(after[1] - target[1]) <= 0.05, detail


def do_execute_trajectory():
    traj_state = get_robot_state()
    start = joints_of(traj_state)
    goal = list(start)
    goal[2] = goal[2] + 0.08
    n_traj = 20
    waypoints = []
    for i in range(n_traj):
        t = float(i + 1) / float(n_traj)
        waypoint = []
        for k in range(7):
            waypoint.append(start[k] + t * (goal[k] - start[k]))
        waypoints.append(waypoint)
    for i in range(8):
        waypoints.append(list(goal))
    trajectory = Trajectory(
        joint_positions=waypoints,
        dt_s=0.05,
        joint_names=names_of(traj_state),
        planner="api_tool_check",
        collision_aware=False,
        expected_start=traj_state,
        arm="primary",
        embodiment="robocasa",
    )
    executed = execute_trajectory(trajectory)
    after = joints_of(get_robot_state())
    detail = dump_execution(executed)
    detail["joint2_before"] = round(start[2], 4)
    detail["joint2_target"] = round(goal[2], 4)
    detail["joint2_after"] = round(after[2], 4)
    detail["n_waypoints"] = len(waypoints)
    return executed.ok and abs(after[2] - goal[2]) <= 0.05, detail


def do_move_to_pose():
    pose = get_robot_state().end_effector_poses["primary"]
    target = {
        "position": [
            float(pose.position[0]),
            float(pose.position[1]),
            float(pose.position[2]) - 0.04,
        ],
        "quaternion_wxyz": [float(v) for v in pose.quaternion_wxyz],
        "frame": pose.frame,
    }
    moved = move_to_pose(target, tolerance=0.03, max_steps=200)
    after = get_robot_state().end_effector_poses["primary"]
    z_error = abs(float(after.position[2]) - target["position"][2])
    detail = dump_execution(moved)
    detail["target"] = target
    detail["ee_before"] = dump_pose(pose)
    detail["ee_after"] = dump_pose(after)
    detail["z_error_m"] = round(z_error, 4)
    return moved.ok and z_error <= 0.04, detail


def do_move_synchronized():
    pose = get_robot_state().end_effector_poses["primary"]
    moved = move_synchronized({"primary": pose}, tolerance=0.03, max_steps=80)
    detail = dump_execution(moved)
    detail["note"] = "unimanual; a typed failure is still a successful probe of this tool"
    return True, detail


call("move_to_joints", do_move_to_joints)
call("execute_trajectory", do_execute_trajectory)
call("move_to_pose", do_move_to_pose)
call("move_synchronized", do_move_synchronized)


# --------------------------------------------------------------------------- base motion: lateral + yaw. Skip forward — spawn faces a counter
# and a blocked drive_base can run ~60 s.

def do_navigate_to_pose():
    before = robocasa.get_base_pose()
    xy_before = [float(before.position[0]), float(before.position[1])]
    lateral = robocasa.navigate_to_pose(0.0, 0.08, 0.0)
    mid = robocasa.get_base_pose()
    xy_mid = [float(mid.position[0]), float(mid.position[1])]
    yaw_before = yaw_of(mid)
    yawed = robocasa.navigate_to_pose(0.0, 0.0, 0.2)
    after = robocasa.get_base_pose()
    yaw_delta = abs(wrap_pi(yaw_of(after) - yaw_before))
    detail = {
        "before": dump_pose(before),
        "after_lateral": dump_pose(mid),
        "after_yaw": dump_pose(after),
        "lateral": dump_execution(lateral),
        "lateral_moved_m": round(dist(xy_mid, xy_before), 4),
        "yaw": dump_execution(yawed),
        "yaw_delta_rad": round(yaw_delta, 4),
        "note": "forward from spawn is usually blocked; not commanded",
    }
    return (lateral.ok and dist(xy_mid, xy_before) > 0.03) or (yawed.ok and yaw_delta > 0.1), detail


call("robocasa.navigate_to_pose", do_navigate_to_pose)


def do_go_home():
    home_joints = hold.get("home_joints")
    if home_joints is None:
        return False, "no captured home"
    homed = go_home()
    after = joints_of(get_robot_state())
    detail = dump_execution(homed)
    detail["joint_error_rad"] = round(dist(after, home_joints), 4)
    detail["joints_after"] = rounds(after, 4)
    return homed.ok or dist(after, home_joints) <= 0.15, detail


call("go_home", do_go_home)


# --------------------------------------------------------------------------- summary

print("")
print("=" * 72)
print("summary")
print("=" * 72)
missing = []
for name in EXPECTED:
    if name not in report:
        missing.append(name)
        print("MISSING  " + name)
    elif report[name]["ok"]:
        print("ok       " + name)
    else:
        print("FAIL     " + name + "  " + str(report[name]["detail"])[:180])
if missing:
    record("coverage", False, missing)
else:
    record("coverage", True, {"n": len(EXPECTED)})

print("")
print("failed: " + str(failed))
print("harness visuals: media/overlays, media/keyframes, media/videos, logs/program.stdout")

result = report

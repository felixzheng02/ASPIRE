# Integration gate for the RoboCasa adapter. NOT a policy for the task.
#
# A real program takes this path -- observation, SAM, back-projection, gripper,
# joint motion, execute_trajectory, go_home, mobile base -- and this file
# *measures* that each step did something. It does not try to open the drawer;
# writing the task program is the agent's job. task_success will still be false.
#
# Sandbox: import math only. ndarray METHODS (points.mean(axis=0), .min, .sum)
# trigger a lazy numpy import and fail with "'mappingproxy' object has no
# attribute '__import__'". Module-level numpy is fine; this file uses lists,
# sum()/len(), and math instead.
import math

report = {}
failed = []


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


# --------------------------------------------------------------------------- identity

context = get_task_context()
instruction = str(context.language)
record(
    "instruction",
    len(instruction.strip()) > 0,
    instruction,
)

observation = get_observation()
cameras = sorted(observation.cameras)
record(
    "cameras",
    "agentview" in observation.cameras and "robot0_eye_in_hand" in observation.cameras,
    cameras,
)

state = get_robot_state()
home_joints = joints_of(state)
record("base_frame", state.base_frame == "robot0_link0", state.base_frame)
record("n_joints", len(home_joints) == 7, len(home_joints))

meta = robocasa.get_controller_metadata()
record("mobile_base", bool(meta["mobile_base"]), meta["mobile_base"])

# --------------------------------------------------------------------------- scene SAM
# "cabinet", not "drawer": ground-truth segmentation resolves whole bodies, and
# RoboCasa's drawers are geometry inside cab_* bodies with no drawer body of
# their own. segment_text(..., "drawer") returning nothing is expected.

found = localize_object("cabinet", camera_name="agentview")
cabinet_mask = None
if found.ok:
    points = found.point_cloud.points
    n_points = int(len(points))
    cloud_frame = found.point_cloud.frame
    centroid = centroid_of(points)
    cabinet_mask = found.segmentation
    record(
        "localize_cabinet",
        n_points >= 20 and cloud_frame == state.base_frame,
        {
            "n_points": n_points,
            "frame": cloud_frame,
            "centroid": [round(v, 3) for v in centroid],
        },
    )
    geometry = estimate_geometry(found.point_cloud)
    extents = [float(v) for v in geometry.extents]
    record(
        "estimate_geometry",
        extents[0] > 0.0 and extents[1] > 0.0 and extents[2] > 0.0,
        {"extents": [round(v, 3) for v in extents], "frame": geometry.frame},
    )
    cropped = crop_point_cloud(
        found.point_cloud,
        [centroid[0] - 2.0, centroid[1] - 2.0, centroid[2] - 2.0],
        [centroid[0] + 2.0, centroid[1] + 2.0, centroid[2] + 2.0],
    )
    record("crop_point_cloud", len(cropped.points) >= 20, int(len(cropped.points)))
else:
    record("localize_cabinet", False, error_text(found.error) or "localize_object failed")

seen = segment_text("agentview", "counter")
if seen.ok and seen.segmentations:
    best = seen.segmentations[0]
    # target_frame must be passed: the API default is the literal "robot_base",
    # which matches no adapter's base frame. localize_object resolves it from
    # robot state; this entry point does not.
    cloud = mask_to_point_cloud(best, "agentview", state.base_frame)
    record(
        "segment_counter",
        len(cloud.points) >= 20 and cloud.frame == state.base_frame,
        {
            "label": best.label,
            "score": round(float(best.score), 3),
            "n_points": int(len(cloud.points)),
            "frame": cloud.frame,
        },
    )
else:
    record("segment_counter", False, error_text(seen.error) or "empty mask")

# --------------------------------------------------------------------------- wrist SAM

wrist = observation.cameras["robot0_eye_in_hand"]
wrist_h = int(wrist.rgb.shape[0])
wrist_w = int(wrist.rgb.shape[1])
wrist_depth = float(wrist.depth_m[wrist_h // 2, wrist_w // 2])
record(
    "wrist_camera",
    wrist_h > 0 and wrist_w > 0 and math.isfinite(wrist_depth) and wrist_depth >= 0.0,
    {"shape": [wrist_h, wrist_w], "center_depth_m": round(wrist_depth, 3)},
)

wrist_prompt = None
for query in ["counter", "cabinet", "robot"]:
    probed = segment_text("robot0_eye_in_hand", query)
    if probed.ok and probed.segmentations:
        wrist_prompt = query
        break
report["wrist_segment_text"] = {
    "ok": wrist_prompt is not None,
    "detail": wrist_prompt or "no fallback prompt hit; not a required check",
}

clicked = segment_points("robot0_eye_in_hand", [[float(wrist_w) / 2.0, float(wrist_h) / 2.0]])
record(
    "segment_points",
    clicked.ok,
    error_text(clicked.error) if not clicked.ok else "point prompt HTTP ok",
)

# --------------------------------------------------------------------------- gripper

closed = close_gripper()
grip_closed = float(get_robot_state().gripper_positions["primary"])
record(
    "close_gripper",
    closed.ok and grip_closed < 0.2,
    {
        "command_ok": closed.ok,
        "measured": round(grip_closed, 3),
        "error": error_text(closed.error),
    },
)

opened = open_gripper()
grip_open = float(get_robot_state().gripper_positions["primary"])
record(
    "open_gripper",
    opened.ok and grip_open > 0.8,
    {
        "command_ok": opened.ok,
        "measured": round(grip_open, 3),
        "error": error_text(opened.error),
    },
)

# --------------------------------------------------------------------------- arm: move_to_joints then execute_trajectory then go_home

before_joints = joints_of(get_robot_state())
target_joints = list(before_joints)
target_joints[1] = target_joints[1] + 0.15
moved = move_to_joints(target_joints, tolerance=0.02, max_steps=120)
after_joints = joints_of(get_robot_state())
record(
    "move_to_joints",
    moved.ok and abs(after_joints[1] - target_joints[1]) <= 0.05,
    {
        "command_ok": moved.ok,
        "joint1_before": round(before_joints[1], 3),
        "joint1_after": round(after_joints[1], 3),
        "joint1_target": round(target_joints[1], 3),
        "error": error_text(moved.error),
    },
)

traj_state = get_robot_state()
traj_start = joints_of(traj_state)
traj_goal = list(traj_start)
traj_goal[2] = traj_goal[2] + 0.10
# execute_trajectory applies one control tick per waypoint (it does not servo).
# Interpolate so the last waypoint is reachable within the adapter's 0.02 rad tolerance.
n_traj = 25
waypoints = []
for i in range(n_traj):
    t = float(i + 1) / float(n_traj)
    waypoint = []
    for k in range(7):
        waypoint.append(traj_start[k] + t * (traj_goal[k] - traj_start[k]))
    waypoints.append(waypoint)
for i in range(10):
    waypoints.append(list(traj_goal))
trajectory = Trajectory(
    joint_positions=waypoints,
    dt_s=0.05,
    joint_names=names_of(traj_state),
    planner="connectivity_check",
    collision_aware=False,
    expected_start=traj_state,
    arm="primary",
    embodiment="robocasa",
)
executed = execute_trajectory(trajectory)
traj_after = joints_of(get_robot_state())
record(
    "execute_trajectory",
    executed.ok and abs(traj_after[2] - traj_goal[2]) <= 0.05,
    {
        "command_ok": executed.ok,
        "joint2_before": round(traj_start[2], 3),
        "joint2_after": round(traj_after[2], 3),
        "joint2_target": round(traj_goal[2], 3),
        "error": error_text(executed.error),
    },
)

homed = go_home()
home_after = joints_of(get_robot_state())
record(
    "go_home",
    homed.ok and dist(home_after, home_joints) <= 0.08,
    {
        "command_ok": homed.ok,
        "joint_error": round(dist(home_after, home_joints), 3),
        "error": error_text(homed.error),
    },
)

# --------------------------------------------------------------------------- base: lateral + yaw. Forward is skipped: a blocked
# drive_base can run up to 1200 ticks (~60 s) and spawn faces a counter.

pose_before = robocasa.get_base_pose()
xy_before = [float(pose_before.position[0]), float(pose_before.position[1])]
lateral = robocasa.navigate_to_pose(0.0, 0.08, 0.0)
pose_mid = robocasa.get_base_pose()
xy_mid = [float(pose_mid.position[0]), float(pose_mid.position[1])]
lateral_moved = dist(xy_mid, xy_before)
record(
    "navigate_lateral",
    lateral.ok and lateral_moved > 0.03,
    {
        "command_ok": lateral.ok,
        "moved_m": round(lateral_moved, 3),
        "error": error_text(lateral.error),
    },
)

yaw_before = yaw_of(pose_mid)
yawed = robocasa.navigate_to_pose(0.0, 0.0, 0.2)
yaw_after = yaw_of(robocasa.get_base_pose())
yaw_delta = abs(wrap_pi(yaw_after - yaw_before))
record(
    "navigate_yaw",
    yawed.ok and yaw_delta > 0.1,
    {
        "command_ok": yawed.ok,
        "yaw_delta_rad": round(yaw_delta, 3),
        "error": error_text(yawed.error),
    },
)

# --------------------------------------------------------------------------- expected failures: IK, grasps, mid-range gripper are
# not wired for RoboCasa. If they start succeeding, that is a regression.

ik = solve_ik(get_robot_state().end_effector_poses["primary"])
record(
    "solve_ik_unsupported",
    not ik.ok,
    error_text(ik.error) or "solve_ik unexpectedly succeeded",
)

if cabinet_mask is None:
    record("generate_grasps_unsupported", False, "no cabinet mask to pass to generate_grasps")
else:
    grasps = generate_grasps("agentview", cabinet_mask)
    record(
        "generate_grasps_unsupported",
        not grasps.ok,
        error_text(grasps.error) or "generate_grasps unexpectedly succeeded",
    )

mid_grip = set_gripper(0.5)
record(
    "set_gripper_mid_unsupported",
    not mid_grip.ok,
    error_text(mid_grip.error) or "set_gripper(0.5) unexpectedly succeeded",
)

# --------------------------------------------------------------------------- gate

for key in sorted(report):
    print(key + ": " + str(report[key]))

result = report
if failed:
    raise Exception("connectivity check failed: " + "; ".join(failed))

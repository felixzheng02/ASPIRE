# Phase 3 camera pose check: localize one object from three wrist-camera views.
#
# Needs SAM3, the bridge, and the object on the table in front of the gripper:
#   .venv-system2/bin/python -m cap_harness.stretch4.run_program --allow-motion \
#       --sam3-url http://127.0.0.1:8114 --set OBJECT="red mug" \
#       --save-images localize_check examples/stretch4/localize_check.py
#
# OBJECT comes from --set. Moves, each one joint from the measured pose:
#   Home -> wrist pitch +0.8 rad (camera about 46 deg down) -> view 1
#   -> lift 10 cm lower -> view 2 -> wrist roll +20 deg -> view 3 -> Home.
# Base, arm and wrist yaw are never changed. If view 1 fails, it goes straight
# back Home. Each view's wrist image and SAM3 mask are saved; check that the
# object is in view and is what was segmented.
# Positions are in odom (metres); z is height above the floor.
import math

LOOK_DOWN_PITCH = 0.8  # rad; URDF convention, positive tilts the camera down
LIFT_DROP = 0.10  # m
ROLL_STEP = math.radians(20.0)
LIFT, PITCH, ROLL = 3, 6, 7  # joint indices in the 8-joint Stretch arm

images = {}
views = {}


def move_one_joint(index, value):
    joints = get_robot_state().joint_positions["primary"].copy()
    joints[index] = value
    return move_to_joints(joints)


def localize(label):
    images[label] = get_observation().cameras["wrist"].rgb
    found = localize_object(OBJECT, camera_name="wrist")
    if not found.ok:
        views[label] = {"ok": False, "error": found.error}
        print(f"{label}: localize_object failed: {found.error}")
        return
    center = [round(float(value), 4) for value in found.geometry.pose.position]
    top_z = round(float(max(found.point_cloud.points[:, 2])), 4)
    images[label + "_mask"] = found.segmentation.mask
    views[label] = {
        "ok": True,
        "center_odom": center,
        "center_z": center[2],
        "top_z": top_z,
        "extents": [round(float(value), 4) for value in found.geometry.extents],
        "points": found.geometry.point_count,
        "sam3_score": round(float(found.segmentation.score), 3),
    }
    print(f"{label}: center (odom) {center}, center z {center[2]} m, top z {top_z} m")


def step(name, outcome):
    steps.append((name, outcome))
    return outcome.ok


steps = []
# Each line runs only if everything before it succeeded. Stop after view 1 if
# localization failed (SAM3 down or object not in view; see its saved image).
if step("home", go_home()) and step("look down", move_one_joint(PITCH, LOOK_DOWN_PITCH)):
    localize("view1_home_lift")
    joints = get_robot_state().joint_positions["primary"]
    if views["view1_home_lift"]["ok"] and step(
        "lift -10 cm", move_one_joint(LIFT, joints[LIFT] - LIFT_DROP)
    ):
        localize("view2_lift_lower")
        joints = get_robot_state().joint_positions["primary"]
        if step("roll +20 deg", move_one_joint(ROLL, joints[ROLL] + ROLL_STEP)):
            localize("view3_roll")
if len(steps) > 1:  # something moved after Home
    step("return home", go_home())

for name, outcome in steps:
    if not outcome.ok:
        print(f"step {name!r} failed: {outcome.error}")

centers = [view["center_odom"] for view in views.values() if view["ok"]]
largest = max(
    [math.dist(a, b) for i, a in enumerate(centers) for b in centers[i + 1 :]], default=None
)
if largest is not None:
    print(f"largest distance between the {len(centers)} estimates: {largest * 100:.1f} cm")
print("heights (compare with the tape-measured object height):")
for label, view in views.items():
    if view["ok"]:
        print(f"  {label}: center z {view['center_z']} m, top z {view['top_z']} m")

result = {
    "object": OBJECT,
    "steps": {name: outcome.ok for name, outcome in steps},
    "views": views,
    "largest_distance_m": largest,
    "images": images,
}

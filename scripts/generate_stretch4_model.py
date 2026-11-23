#!/usr/bin/env python3
"""Generate the Stretch 4 kinematic MuJoCo model used by cap_harness.stretch4.

Run once per calibration with the lab's Stretch venv (it has Hello Robot's
``hello-robot-stretch4-urdf`` package and xacro) and ROS sourced:

    source /opt/ros/jazzy/setup.bash
    /opt/rail/stretch4/venv/bin/python scripts/generate_stretch4_model.py --nominal

``--nominal`` uses Hello Robot's nominal URDF. stretch-se4-4037 has no
``stretch_calibration_values.yaml`` (the robot itself runs on nominal values),
so the committed model is nominal. If a calibration file appears, pass its
fleet directory instead (for example ``~/stretch_user/stretch-se4-4037``) and
the package's ``apply_calibration_to_urdf`` is applied. The script builds the
SE4 URDF with the PG4 tool and writes
``src/cap_harness/stretch4/description/stretch4.xml``. Commit that file;
runtime does not need the URDF package.

The model keeps only the chain to the grasp center and the wrist camera. The
base is three planar joints (``base_x``, ``base_y``, ``base_theta``) in odom.
The four telescoping arm joints become one ``arm`` joint with their summed
travel; that is exact when the segment axes are parallel, which is checked.
No meshes or collision geometry: this model is for FK and IK only.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

MODEL = "SE4"
BATCH = "francis"
TOOL = "eoa_wrist_dw4_tool_pg4"
OUTPUT = Path(__file__).resolve().parents[1] / "src/cap_harness/stretch4/description/stretch4.xml"
# Frames ASPIRE needs: the grasp site and the wrist RGB-D optical frame (the
# wrist RGB is the right stereo camera, CAM_C, and depth is aligned to it).
SITES = {
    "grasp_center_link": "grasp_site",
    "gripper_right_camera_color_optical_frame": "wrist_camera_optical",
}
# URDF joint -> model joint name.
JOINTS = {
    "lift_joint": "lift",
    "arm_l1_joint": "arm",
    "wrist_yaw_joint": "wrist_yaw",
    "wrist_pitch_joint": "wrist_pitch",
    "wrist_roll_joint": "wrist_roll",
}
ARM_SEGMENTS = ("arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint")
# Largest segment rotation the arm merge tolerates (rad); 1e-3 rad over the
# 0.52 m arm is about 0.5 mm.
_MERGE_TOLERANCE_RAD = 1e-3
_MASS = "0.01"
_INERTIA = "1e-5 1e-5 1e-5"


def calibration_note(calibration: Path) -> str:
    """Hash and joint count of the calibration file, for the model header."""
    import yaml

    data = yaml.safe_load(calibration.read_text()) or {}
    joints = data.get("joint_calibration") or data.get("robot_calibration") or {}
    digest = hashlib.sha256(calibration.read_bytes()).hexdigest()[:12]
    return f"{calibration.parent.name}/{calibration.name} sha256 {digest}, {len(joints)} calibrated joints"


def robot_urdf(fleet_dir: Path | None) -> str:
    """Nominal URDF when ``fleet_dir`` is None, else with its calibration applied."""
    from stretch4_urdf.utils.calibration_utils import apply_calibration_to_urdf
    from stretch4_urdf.utils.urdf_utils_generate_from_base_xacro import (
        generate_urdf_from_xacro,
    )

    if fleet_dir is None:
        return generate_urdf_from_xacro(MODEL, BATCH, TOOL)
    calibration = fleet_dir / "stretch_calibration_values.yaml"
    if not calibration.is_file():
        # apply_calibration_to_urdf silently falls back to nominal values.
        sys.exit(f"no calibration file at {calibration}; use --nominal for nominal values")
    # Set only after import: with these set, importing the package also loads
    # the robot's stretch_body YAML files, which this script does not need.
    os.environ["HELLO_FLEET_PATH"] = str(fleet_dir.parent)
    os.environ["HELLO_FLEET_ID"] = fleet_dir.name
    return apply_calibration_to_urdf(generate_urdf_from_xacro(MODEL, BATCH, TOOL))


def _floats(text: str | None, default: str = "0 0 0") -> list[float]:
    return [float(value) for value in (text or default).split()]


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """URDF fixed-axis rpy to a wxyz quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _fmt(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def build_mjcf(urdf: str, note: str) -> tuple[str, dict[str, tuple[float, float]]]:
    robot = ET.fromstring(urdf)
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    parent_joint = {joint.find("child").get("link"): joint for joint in joints.values()}

    # Links on the chains to the needed frames.
    keep: set[str] = set()
    for frame in SITES:
        link = frame
        while link in parent_joint:
            keep.add(link)
            link = parent_joint[link].find("parent").get("link")
        keep.add(link)
    root = "base_footprint"
    if root not in keep:
        sys.exit("URDF chain does not start at base_footprint")

    for name in ARM_SEGMENTS[1:]:
        rpy = _floats(joints[name].find("origin").get("rpy"))
        axis = _floats(joints[name].find("axis").get("xyz"))
        if max(abs(value) for value in rpy) > _MERGE_TOLERANCE_RAD or axis != [1.0, 0.0, 0.0]:
            sys.exit(f"{name} is not parallel to arm_l1_joint; cannot merge arm segments")

    limits: dict[str, tuple[float, float]] = {}
    arm_upper = 0.0
    for name in ARM_SEGMENTS:
        limit = joints[name].find("limit")
        arm_upper += float(limit.get("upper")) - float(limit.get("lower"))

    mujoco = ET.Element("mujoco", model="stretch4")
    ET.SubElement(mujoco, "compiler", angle="radian")
    ET.SubElement(mujoco, "option", gravity="0 0 0")
    worldbody = ET.SubElement(mujoco, "worldbody")
    base = ET.SubElement(worldbody, "body", name=root)
    ET.SubElement(base, "inertial", pos="0 0 0", mass=_MASS, diaginertia=_INERTIA)
    ET.SubElement(base, "joint", name="base_x", type="slide", axis="1 0 0")
    ET.SubElement(base, "joint", name="base_y", type="slide", axis="0 1 0")
    ET.SubElement(base, "joint", name="base_theta", type="hinge", axis="0 0 1")

    def add_children(link: str, element: ET.Element) -> None:
        for joint in joints.values():
            if joint.find("parent").get("link") != link:
                continue
            child = joint.find("child").get("link")
            if child not in keep:
                continue
            origin = joint.find("origin")
            pos = _floats(origin.get("xyz") if origin is not None else None)
            quat = rpy_to_quat(*_floats(origin.get("rpy") if origin is not None else None))
            body = ET.SubElement(element, "body", name=child, pos=_fmt(pos), quat=_fmt(quat))
            ET.SubElement(body, "inertial", pos="0 0 0", mass=_MASS, diaginertia=_INERTIA)
            name = joint.get("name")
            if name in JOINTS:
                limit = joint.find("limit")
                lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                if name == "arm_l1_joint":
                    lower, upper = 0.0, arm_upper
                limits[JOINTS[name]] = (lower, upper)
                ET.SubElement(
                    body,
                    "joint",
                    name=JOINTS[name],
                    type="slide" if joint.get("type") == "prismatic" else "hinge",
                    axis=_fmt(_floats(joint.find("axis").get("xyz"))),
                    range=_fmt([lower, upper]),
                    limited="true",
                )
            elif joint.get("type") != "fixed" and name not in ARM_SEGMENTS:
                sys.exit(f"unexpected movable joint {name} on the kept chain")
            if child in SITES:
                ET.SubElement(body, "site", name=SITES[child], size="0.005")
            add_children(child, body)

    add_children(root, base)
    ET.indent(mujoco)
    header = (
        f"<!-- Generated by scripts/generate_stretch4_model.py from {MODEL}_{BATCH} + {TOOL},"
        f" calibration {note}. Do not edit. -->\n"
    )
    return header + ET.tostring(mujoco, encoding="unicode") + "\n", limits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "fleet_dir", type=Path, nargs="?", help="directory with stretch_calibration_values.yaml"
    )
    source.add_argument("--nominal", action="store_true", help="use nominal URDF values")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.nominal:
        xml, limits = build_mjcf(robot_urdf(None), "none (nominal URDF values)")
    else:
        fleet_dir = args.fleet_dir.expanduser().resolve()
        urdf = robot_urdf(fleet_dir)
        xml, limits = build_mjcf(
            urdf, calibration_note(fleet_dir / "stretch_calibration_values.yaml")
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml)
    print(f"wrote {args.output}")
    print("joint limits (copy into agentic_robotics/aspire_bridge.py JOINT_LIMITS):")
    for name, (lower, upper) in limits.items():
        print(f"    {name!r}: ({lower:.6g}, {upper:.6g}),")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

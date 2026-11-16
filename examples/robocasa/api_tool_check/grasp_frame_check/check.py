"""A/B check: is a GraspCandidate pose panda_hand, while move_to_pose wants EEF?

Writes only under this directory. Does not touch the parent tool-check dump.

The claim has two halves that have to be true together:

1. ``get_robot_state`` / ``move_to_pose`` / ``solve_ik`` speak ``gripper0_right_eef``.
   PyRoki converts that to ``panda_hand`` at the provider boundary
   (``position + R @ [0, 0, -0.1]``, and on robocasa ``R @ Rz(+90 deg)``).
2. ``generate_grasps`` publishes a ``panda_hand`` pose in ``robot0_link0``
   (origin ~10 cm behind ``contact_point`` along local +Z; CGN client already
   right-multiplies ``Rz(+90 deg)``).

A previous probe drove only the *converted* pose, so it could not A/B. This
script does both, plus an identity check that does not depend on GraspNet.

Usage (from ``aspire/``)::

    PYTHONPATH="" .venv-robocasa/bin/python \\
        examples/robocasa/api_tool_check/grasp_frame_check/check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
OUT = HERE / "out"
sys.path.insert(0, str(PARENT))

from cap_harness.contracts import Pose  # noqa: E402
from cap_harness.geometry import matrix_to_pose  # noqa: E402
from session import (  # noqa: E402
    NumpyJson,
    build_session,
    dump_execution,
    dump_pose,
    error_text,
    pose_matrix,
    project_points,
    quat_to_matrix,
    quaternion_angle_deg,
)

# Exact constants from cap_harness.providers.pyroki.client.
PANDA_EEF_TO_HAND_OFFSET_M = np.array([0.0, 0.0, -0.1], dtype=np.float64)
RZ_PLUS_90 = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
RZ_MINUS_90 = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
BASE_FRAME = "robot0_link0"
NEAR_M = 0.03
ROT_MATCH_DEG = 15.0
ROT_ORTHO_DEG = (75.0, 105.0)


def eef_to_hand(pose: Pose) -> Pose:
    """Public EEF -> panda_hand, same formula PyRoki uses for robocasa."""
    rotation = quat_to_matrix(pose.quaternion_wxyz)
    transform = np.eye(4)
    transform[:3, :3] = rotation @ RZ_PLUS_90
    transform[:3, 3] = np.asarray(pose.position, dtype=np.float64) + rotation @ PANDA_EEF_TO_HAND_OFFSET_M
    return matrix_to_pose(transform, frame=pose.frame)


def hand_to_eef(pose: Pose) -> Pose:
    """Inverse of ``eef_to_hand`` (Z is shared, so +0.1 m along local +Z)."""
    rotation = quat_to_matrix(pose.quaternion_wxyz)
    transform = np.eye(4)
    transform[:3, :3] = rotation @ RZ_MINUS_90
    transform[:3, 3] = np.asarray(pose.position, dtype=np.float64) + rotation[:, 2] * 0.1
    return matrix_to_pose(transform, frame=pose.frame)


def with_rotation(pose: Pose, rotation: np.ndarray) -> Pose:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(pose.position, dtype=np.float64)
    return matrix_to_pose(transform, frame=pose.frame)


def _xyz(pose: Pose) -> np.ndarray:
    return np.asarray(pose.position, dtype=np.float64)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _between(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def _save_png(name: str, rgb: np.ndarray) -> str:
    from PIL import Image

    path = OUT / name
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
    return path.name


def _overlay(
    name: str,
    rgb: np.ndarray,
    camera: Any,
    marks: list[tuple[str, np.ndarray, tuple[int, int, int]]],
) -> str:
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8).copy())
    draw = ImageDraw.Draw(image)
    h, w = rgb.shape[:2]
    legend_y = 8
    for label, xyz, color in marks:
        pixels = project_points(np.asarray(xyz, dtype=np.float64).reshape(1, 3), camera)[0]
        u, v, depth = float(pixels[0]), float(pixels[1]), float(pixels[2])
        visible = np.isfinite(u) and np.isfinite(v) and depth > 0 and 0 <= u < w and 0 <= v < h
        if visible:
            x, y = int(round(u)), int(round(v))
            r = 9
            draw.line((x - r, y, x + r, y), fill=color, width=3)
            draw.line((x, y - r, x, y + r), fill=color, width=3)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.rectangle((8, legend_y, 20, legend_y + 12), fill=color)
        tag = label if visible else f"{label} (off-image)"
        draw.text((26, legend_y), tag, fill=color)
        legend_y += 16
    path = OUT / name
    image.save(path)
    return path.name


def _pose_record(pose: Pose) -> dict[str, Any]:
    matrix = pose_matrix(pose)
    return {**dump_pose(pose), "local_z_base": np.round(matrix[:3, 2], 5).tolist()}


def _ik(tools: Any, target: Pose) -> dict[str, Any]:
    result = tools["solve_ik"](target)
    pos_err = result.diagnostics.get("position_error_m") if result.ok else None
    ori_err = result.diagnostics.get("orientation_error_rad") if result.ok else None
    return {
        "ok": bool(result.ok),
        "error": error_text(result.error),
        "diagnostics": {k: str(v) for k, v in dict(result.diagnostics).items()},
        "position_error_m": None if pos_err is None else float(pos_err),
        "orientation_error_rad": None if ori_err is None else float(ori_err),
        "joint_positions_rad": None
        if result.joint_positions is None
        else np.round(np.asarray(result.joint_positions, dtype=np.float64), 5).tolist(),
        "joints": None
        if result.joint_positions is None
        else np.asarray(result.joint_positions, dtype=np.float64),
    }


def _move_pose(tools: Any, target: Pose, *, max_steps: int = 200) -> dict[str, Any]:
    moved = tools["move_to_pose"](target, 0.015, max_steps)
    reached = tools["get_robot_state"]().end_effector_poses["primary"]
    return {
        "ok": bool(moved.ok),
        "how": "move_to_pose",
        "execution": dump_execution(moved),
        "target": _pose_record(target),
        "reached": _pose_record(reached),
        "position_error_m": round(_dist(_xyz(reached), _xyz(target)), 5),
        "orientation_error_deg": round(
            quaternion_angle_deg(reached.quaternion_wxyz, target.quaternion_wxyz), 3
        ),
        "reached_pose": reached,
    }


def _move_joints(tools: Any, joints: np.ndarray, *, max_steps: int = 240) -> dict[str, Any]:
    moved = tools["move_to_joints"](joints, 0.02, max_steps)
    reached = tools["get_robot_state"]().end_effector_poses["primary"]
    reached_joints = np.asarray(
        tools["get_robot_state"]().joint_positions["primary"], dtype=np.float64
    )
    return {
        "ok": bool(moved.ok),
        "how": "move_to_joints(solve_ik(...))",
        "execution": dump_execution(moved),
        "reached": _pose_record(reached),
        "max_joint_error_rad": round(float(np.max(np.abs(reached_joints - joints))), 5),
        "reached_pose": reached,
    }


def _pick_grasp(tools: Any) -> tuple[Any, Any, list[int]]:
    found = tools["segment_text"]("agentview", "fish")
    if not found.ok or not found.segmentations:
        return None, None, []
    segmentation = found.segmentations[0]
    attempts: list[int] = []
    best = None
    for _ in range(8):
        grasps = tools["generate_grasps"]("agentview", segmentation, max_candidates=8)
        n = len(grasps.grasps) if grasps.ok else 0
        attempts.append(n)
        if grasps.ok:
            best = tools["select_grasp"](grasps, "highest_score")
            if best is not None:
                return segmentation, best, attempts
    return segmentation, None, attempts


def _grasp_marks(
    origin: np.ndarray, contact: np.ndarray, converted: Pose, reached: Pose
) -> list[tuple[str, np.ndarray, tuple[int, int, int]]]:
    return [
        ("grasp origin (red)", origin, (209, 73, 91)),
        ("contact (green)", contact, (46, 160, 67)),
        ("converted EEF target (yellow)", _xyz(converted), (218, 165, 32)),
        ("measured EEF (blue)", _xyz(reached), (46, 134, 171)),
    ]


def _drive(
    session: Any,
    tools: Any,
    *,
    label: str,
    target: Pose,
    origin: np.ndarray,
    contact: np.ndarray,
    converted: Pose,
    grasp_pose: Pose,
    joints: np.ndarray | None,
) -> dict[str, Any]:
    session.reset()
    tools["open_gripper"]()
    local_z = pose_matrix(target)[:3, 2]
    standoff = Pose(
        position=_xyz(target) - 0.08 * local_z,
        quaternion_wxyz=np.asarray(target.quaternion_wxyz, dtype=np.float64),
        frame=target.frame,
    )
    standoff_move = _move_pose(tools, standoff, max_steps=200)
    standoff_move.pop("reached_pose", None)
    if joints is not None:
        result = _move_joints(tools, joints)
    else:
        result = _move_pose(tools, target)
    reached = result.pop("reached_pose")
    obs = tools["get_observation"]()
    result["standoff"] = standoff_move
    result["target"] = _pose_record(target)
    result["reached_to_grasp_origin_m"] = round(_dist(_xyz(reached), origin), 5)
    result["reached_to_contact_m"] = round(_dist(_xyz(reached), contact), 5)
    result["reached_to_converted_m"] = round(_dist(_xyz(reached), _xyz(converted)), 5)
    result["reached_vs_grasp_deg"] = round(
        quaternion_angle_deg(reached.quaternion_wxyz, grasp_pose.quaternion_wxyz), 3
    )
    result["reached_vs_converted_deg"] = round(
        quaternion_angle_deg(reached.quaternion_wxyz, converted.quaternion_wxyz), 3
    )
    marks = _grasp_marks(origin, contact, converted, reached)
    result["overlay_agentview"] = _overlay(
        f"overlay_{label}_agentview.png",
        obs.cameras["agentview"].rgb,
        obs.cameras["agentview"],
        marks,
    )
    result["overlay_wrist"] = _overlay(
        f"overlay_{label}_wrist.png",
        obs.cameras["robot0_eye_in_hand"].rgb,
        obs.cameras["robot0_eye_in_hand"],
        marks,
    )
    _save_png(f"after_{label}_agentview.png", obs.cameras["agentview"].rgb)
    _save_png(f"after_{label}_wrist.png", obs.cameras["robot0_eye_in_hand"].rgb)
    return result


def _write_results(payload: dict[str, Any]) -> None:
    mapping = payload["mapping"]
    identity = payload["identity"]
    grasp = payload.get("grasp")
    v = payload["verdicts"]

    mapping_ok = mapping["roundtrip_position_error_m"] < 1e-9 and mapping["eef_to_hand_delta_m"] == 0.1
    stay = v["identity_stay"]
    roll = v["identity_roll90"]
    geom = v["grasp_geometry_is_panda_hand"]
    ik_prefers_converted = v["ik_prefers_converted_eef"]
    converted_nearer = v["converted_drive_nearer_contact_than_raw"]
    converted_yaw = v["converted_drive_yaw_is_90_from_grasp"]

    if mapping_ok and stay and geom and ik_prefers_converted and converted_nearer and converted_yaw:
        headline = "REAL"
        detail = (
            "Grasp poses are panda_hand; move_to_pose / solve_ik take gripper0_right_eef. "
            "The 10 cm + Rz(-90°) map is the missing conversion."
        )
    elif mapping_ok and stay and geom and ik_prefers_converted:
        headline = "REAL (geometry + IK; Cartesian drive messy)"
        detail = (
            "Grasp origin sits 10 cm behind contact along local +Z, and PyRoki's residual is "
            "orders of magnitude smaller on the converted EEF than on the raw grasp. "
            "Driving both poses to arrival was not clean enough to be the sole proof."
        )
    elif mapping_ok and stay and geom:
        headline = "LIKELY REAL (geometry only)"
        detail = (
            "The grasp origin is a panda_hand pattern (10 cm along +Z from contact, 90° from "
            "the EEF map), and move_to_pose tracks the reported EEF. IK/drive did not add a "
            "clean A/B landing."
        )
    else:
        headline = "NOT CONFIRMED"
        detail = "One of the mapping / EEF-tracking / grasp-geometry checks did not hold."

    def mm(value: float | None) -> str:
        return "n/a" if value is None else f"{1000 * float(value):.1f} mm"

    def deg(value: float | None) -> str:
        return "n/a" if value is None else f"{float(value):.1f}°"

    lines = [
        "# Grasp frame check",
        "",
        f"**Verdict: {headline}**",
        "",
        detail,
        "",
        "Task `PickPlaceCounterToStove`, seed 101. Outputs stay in this directory.",
        "",
        "## 1. The two frames, as algebra on the spawn EEF",
        "",
        "PyRoki's robocasa map (copied, not imported) from the measured EEF:",
        "",
        f"- translation: **{mapping['eef_to_hand_delta_m']*100:.1f} cm** "
        f"(local +Z of the EEF at spawn is world −Z, so the hand sits above the fingertips)",
        f"- rotation: **{mapping['eef_to_hand_rot_deg']:.1f}°** (local Rz(+90°))",
        f"- inverse roundtrip: {mapping['roundtrip_position_error_m']:.2e} m, "
        f"{mapping['roundtrip_orientation_error_deg']:.4f}°",
        "",
        "## 2. Does move_to_pose track gripper0_right_eef?",
        "",
        "| command | expected if it wants EEF | measured |",
        "| --- | --- | --- |",
        f"| current EEF | stay | {mm(identity['command_current_eef']['position_error_m'])}, "
        f"{deg(identity['command_current_eef']['orientation_error_deg'])}; "
        f"{'pass' if stay else 'FAIL'} |",
        f"| same position, panda_hand quaternion (Rz +90°) | in-place 90° roll | "
        f"shift {mm(identity['command_roll90']['measured_shift_from_spawn_m'])}, "
        f"yaw {deg(identity['command_roll90']['measured_rot_from_spawn_deg'])}, "
        f"ok={identity['command_roll90']['ok']}; "
        f"{'pass' if roll else 'FAIL'} |",
        f"| panda_hand pose as if it were EEF | unreachable from spawn (+Z) or 10 cm + 90° | "
        f"{identity['command_hand_as_eef']['execution']['error'] or 'moved'} |",
        "",
        "The last row is **not** a disproof: from spawn, `eef_to_hand` is +10 cm world-Z, "
        "and `solve_ik` already refuses +3 cm of +Z. The roll-only row isolates the 90°.",
        "",
        f"Wrist projection of the reported EEF: `{identity['wrist_eef_px']}` "
        "(should sit between the two visible fingertips). Predicted panda_hand: "
        f"`{identity['wrist_hand_px']}` (behind the camera / in the gripper body).",
        "",
    ]

    if grasp is None:
        lines.append("No grasp candidate this run; grasp half skipped.")
    else:
        geom_d = grasp["geometry"]
        raw = grasp.get("drive_raw")
        conv = grasp.get("drive_converted")
        lines.extend(
            [
                "## 3. Grasp geometry (no motion)",
                "",
                f"- generate_grasps retries: `{grasp['attempts']}`, score `{geom_d['score']}`",
                f"- pose → contact: **{geom_d['pose_to_contact_m']:.4f} m** "
                f"(cos to local +Z = {geom_d['cos_local_z_to_contact']:.4f})",
                f"- converted EEF → contact: **{geom_d['converted_to_contact_m']:.4f} m**",
                f"- converted − origin: {geom_d['converted_minus_origin_m']:.4f} m",
                f"- grasp vs converted orientation: **{geom_d['grasp_vs_converted_rot_deg']:.1f}°**",
                f"- nearest cloud point to pose origin: {geom_d['nearest_cloud_to_pose_m']:.4f} m "
                f"(the origin is not on the object)",
                f"- nearest cloud point to contact: {geom_d['nearest_cloud_to_contact_m']:.4f} m",
                "",
                "Contact-GraspNet's Panda gripper has its origin ~10.34 cm behind the fingertips. "
                "If the published pose were already `gripper0_right_eef`, contact would sit on the "
                "origin, not 10 cm along +Z.",
                "",
                "## 4. solve_ik residuals (no motion)",
                "",
                "PyRoki maps the public pose to `panda_hand` *again*. If the grasp is already "
                "panda_hand, the converted EEF is the pose it actually wants, and the raw grasp "
                "is a double-mapped leftover.",
                "",
                "| target | ok | PyRoki position residual | orientation residual |",
                "| --- | --- | ---: | ---: |",
                f"| raw grasp pose | {grasp['ik_raw']['ok']} | "
                f"{mm(grasp['ik_raw']['position_error_m'])} | "
                f"{grasp['ik_raw']['orientation_error_rad']} rad |",
                f"| converted EEF | {grasp['ik_converted']['ok']} | "
                f"{mm(grasp['ik_converted']['position_error_m'])} | "
                f"{grasp['ik_converted']['orientation_error_rad']} rad |",
                "",
                f"IK prefers converted: **{'yes' if ik_prefers_converted else 'no'}**.",
                "",
                "## 5. Drive A/B (reset between trials; standoff then IK joints)",
                "",
            ]
        )
        if raw and conv:
            lines.extend(
                [
                    "| drive | EEF vs grasp origin | EEF vs contact | yaw vs grasp | yaw vs converted | arrived? |",
                    "| --- | ---: | ---: | ---: | ---: | --- |",
                    f"| raw grasp | {mm(raw['reached_to_grasp_origin_m'])} | "
                    f"{mm(raw['reached_to_contact_m'])} | {deg(raw['reached_vs_grasp_deg'])} | "
                    f"{deg(raw['reached_vs_converted_deg'])} | {raw['ok']} / {raw['how']} |",
                    f"| converted EEF | {mm(conv['reached_to_grasp_origin_m'])} | "
                    f"{mm(conv['reached_to_contact_m'])} | {deg(conv['reached_vs_grasp_deg'])} | "
                    f"{deg(conv['reached_vs_converted_deg'])} | {conv['ok']} / {conv['how']} |",
                    "",
                    "If the claim is true: raw puts the *fingertips* where the gripper *base* "
                    "should be (often the stove, 10 cm short of the fish, and a hard reach). "
                    "Converted puts the fingertips on contact. Wrist images: "
                    "`after_raw_grasp_wrist.png` vs `after_converted_grasp_wrist.png`.",
                    "",
                    f"Converted drive nearer contact than raw: "
                    f"**{'yes' if converted_nearer else 'no'}**. "
                    f"Converted yaw vs grasp ~90°: **{'yes' if converted_yaw else 'no'}**.",
                    "",
                ]
            )
        lines.append(
            "Overlays: red = grasp origin, green = contact, yellow = converted EEF, blue = measured EEF."
        )

    lines.extend(
        [
            "",
            "## Code facts this run is checking, not assuming",
            "",
            "- Simulator `eef_pos` is body `gripper0_right_eef` (`robocasa_env.py`).",
            "- PyRoki/cuRobo map that public pose to `panda_hand` with a 10 cm local-Z offset "
            "and, on robocasa only, `Rz(+90°)`.",
            "- Contact-GraspNet client right-multiplies each 4×4 by the same `Rz(+90°)` "
            "(`orientation_correction` in grasp metadata) and leaves translation as CGN returned it.",
            "- `GraspCandidate.pose.frame` is `robot0_link0` (parent frame), not a link name.",
            "",
        ]
    )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["headline"] = headline
    payload["headline_detail"] = detail


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    session = build_session("PickPlaceCounterToStove", 101)
    tools = session.tools
    payload: dict[str, Any] = {
        "task": "PickPlaceCounterToStove",
        "seed": 101,
        "public_tool_count": len(tools),
    }
    try:
        session.reset()
        spawn = tools["get_robot_state"]().end_effector_poses["primary"]
        hand = eef_to_hand(spawn)
        back = hand_to_eef(hand)
        payload["mapping"] = {
            "spawn_eef": _pose_record(spawn),
            "mapped_panda_hand": _pose_record(hand),
            "hand_to_eef_roundtrip": _pose_record(back),
            "eef_to_hand_delta_m": round(_dist(_xyz(hand), _xyz(spawn)), 5),
            "eef_to_hand_rot_deg": round(
                quaternion_angle_deg(hand.quaternion_wxyz, spawn.quaternion_wxyz), 3
            ),
            "roundtrip_position_error_m": round(_dist(_xyz(back), _xyz(spawn)), 9),
            "roundtrip_orientation_error_deg": round(
                quaternion_angle_deg(back.quaternion_wxyz, spawn.quaternion_wxyz), 6
            ),
        }

        observation = tools["get_observation"]()
        _save_png("spawn_agentview.png", observation.cameras["agentview"].rgb)
        _save_png("spawn_wrist.png", observation.cameras["robot0_eye_in_hand"].rgb)
        wrist = observation.cameras["robot0_eye_in_hand"]
        eef_px = project_points(_xyz(spawn).reshape(1, 3), wrist)[0]
        hand_px = project_points(_xyz(hand).reshape(1, 3), wrist)[0]
        _overlay(
            "overlay_spawn_wrist.png",
            wrist.rgb,
            wrist,
            [
                ("measured EEF (blue)", _xyz(spawn), (46, 134, 171)),
                ("predicted panda_hand (red)", _xyz(hand), (209, 73, 91)),
            ],
        )

        identity_move = _move_pose(tools, spawn)
        identity_move.pop("reached_pose", None)

        session.reset()
        spawn = tools["get_robot_state"]().end_effector_poses["primary"]
        hand = eef_to_hand(spawn)
        roll_target = with_rotation(spawn, quat_to_matrix(hand.quaternion_wxyz))
        roll_ik = _ik(tools, roll_target)
        roll_move = _move_pose(tools, roll_target)
        reached = roll_move.pop("reached_pose")
        roll_move["measured_shift_from_spawn_m"] = round(_dist(_xyz(reached), _xyz(spawn)), 5)
        roll_move["measured_rot_from_spawn_deg"] = round(
            quaternion_angle_deg(reached.quaternion_wxyz, spawn.quaternion_wxyz), 3
        )
        roll_move["ik"] = {k: v for k, v in roll_ik.items() if k != "joints"}
        _save_png(
            "after_roll90_wrist.png",
            tools["get_observation"]().cameras["robot0_eye_in_hand"].rgb,
        )
        _save_png(
            "after_roll90_agentview.png",
            tools["get_observation"]().cameras["agentview"].rgb,
        )

        session.reset()
        spawn = tools["get_robot_state"]().end_effector_poses["primary"]
        hand = eef_to_hand(spawn)
        hand_ik = _ik(tools, hand)
        hand_move = _move_pose(tools, hand)
        hand_move.pop("reached_pose", None)
        hand_move["ik"] = {k: v for k, v in hand_ik.items() if k != "joints"}

        payload["identity"] = {
            "command_current_eef": identity_move,
            "command_roll90": roll_move,
            "command_hand_as_eef": hand_move,
            "wrist_eef_px": np.round(eef_px[:2], 1).tolist(),
            "wrist_eef_depth_m": round(float(eef_px[2]), 4),
            "wrist_hand_px": np.round(hand_px[:2], 1).tolist(),
            "wrist_hand_depth_m": round(float(hand_px[2]), 4),
        }

        session.reset()
        observation = tools["get_observation"]()
        _save_png("grasp_scene.png", observation.cameras["agentview"].rgb)
        segmentation, best, attempts = _pick_grasp(tools)
        verdicts = {
            "identity_stay": identity_move["position_error_m"] < NEAR_M
            and identity_move["orientation_error_deg"] < ROT_MATCH_DEG,
            "identity_roll90": bool(roll_move["ok"])
            and _between(roll_move["measured_rot_from_spawn_deg"], *ROT_ORTHO_DEG)
            and roll_move["measured_shift_from_spawn_m"] < 0.05,
            "grasp_geometry_is_panda_hand": False,
            "ik_prefers_converted_eef": False,
            "converted_drive_nearer_contact_than_raw": False,
            "converted_drive_yaw_is_90_from_grasp": False,
        }

        if best is None:
            payload["grasp"] = None
            payload["grasp_notes"] = (
                f"no grasp after retries {attempts}; segmentation={segmentation is not None}"
            )
        else:
            grasp_pose = best.pose
            converted = hand_to_eef(grasp_pose)
            contact = np.asarray(best.metadata["contact_point"], dtype=np.float64)
            origin = _xyz(grasp_pose)
            delta = contact - origin
            local_z = pose_matrix(grasp_pose)[:3, 2]
            cloud = tools["mask_to_point_cloud"](segmentation, "agentview", BASE_FRAME)
            geom = {
                "score": round(float(best.score), 4),
                "width_m": float(best.width_m),
                "approach_distance_m": float(best.approach_distance_m),
                "metadata": {k: str(v) for k, v in dict(best.metadata).items()},
                "grasp_pose": _pose_record(grasp_pose),
                "converted_eef_pose": _pose_record(converted),
                "contact_point": np.round(contact, 5).tolist(),
                "pose_to_contact_m": round(float(np.linalg.norm(delta)), 5),
                "cos_local_z_to_contact": round(
                    float(np.dot(local_z, delta) / (np.linalg.norm(delta) + 1e-12)), 5
                ),
                "grasp_vs_converted_rot_deg": round(
                    quaternion_angle_deg(grasp_pose.quaternion_wxyz, converted.quaternion_wxyz), 3
                ),
                "converted_minus_origin_m": round(_dist(_xyz(converted), origin), 5),
                "converted_to_contact_m": round(_dist(_xyz(converted), contact), 5),
                "nearest_cloud_to_pose_m": round(
                    float(np.min(np.linalg.norm(cloud.points - origin, axis=1))), 4
                ),
                "nearest_cloud_to_contact_m": round(
                    float(np.min(np.linalg.norm(cloud.points - contact, axis=1))), 4
                ),
            }
            ik_raw = _ik(tools, grasp_pose)
            ik_converted = _ik(tools, converted)
            raw_res = ik_raw["position_error_m"]
            conv_res = ik_converted["position_error_m"]
            drive_raw = _drive(
                session,
                tools,
                label="raw_grasp",
                target=grasp_pose,
                origin=origin,
                contact=contact,
                converted=converted,
                grasp_pose=grasp_pose,
                joints=ik_raw["joints"],
            )
            drive_converted = _drive(
                session,
                tools,
                label="converted_grasp",
                target=converted,
                origin=origin,
                contact=contact,
                converted=converted,
                grasp_pose=grasp_pose,
                joints=ik_converted["joints"],
            )
            payload["grasp"] = {
                "attempts": attempts,
                "geometry": geom,
                "ik_raw": {k: v for k, v in ik_raw.items() if k != "joints"},
                "ik_converted": {k: v for k, v in ik_converted.items() if k != "joints"},
                "drive_raw": drive_raw,
                "drive_converted": drive_converted,
            }
            verdicts["grasp_geometry_is_panda_hand"] = (
                _between(geom["pose_to_contact_m"], 0.08, 0.12)
                and abs(geom["cos_local_z_to_contact"]) > 0.95
                and geom["converted_to_contact_m"] < 0.02
                and abs(geom["grasp_vs_converted_rot_deg"] - 90.0) < 1.0
            )
            verdicts["ik_prefers_converted_eef"] = (
                bool(ik_converted["ok"])
                and conv_res is not None
                and (raw_res is None or conv_res < 0.1 * max(raw_res, 1e-6) or conv_res < 1e-4)
            )
            verdicts["converted_drive_nearer_contact_than_raw"] = (
                drive_converted["reached_to_contact_m"] < drive_raw["reached_to_contact_m"]
                and drive_converted["reached_to_contact_m"]
                < drive_converted["reached_to_grasp_origin_m"]
            )
            verdicts["converted_drive_yaw_is_90_from_grasp"] = _between(
                drive_converted["reached_vs_grasp_deg"], *ROT_ORTHO_DEG
            )

        payload["verdicts"] = verdicts
        _write_results(payload)
        (OUT / "records.json").write_text(
            json.dumps(payload, cls=NumpyJson, indent=2), encoding="utf-8"
        )
        print(payload["headline"])
        print(payload["headline_detail"])
        print(f"wrote {OUT / 'RESULTS.md'}")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())

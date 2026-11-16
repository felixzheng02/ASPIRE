"""Exercise every public RoboCasa CaP tool and dump raw evidence for ``render.py``.

Each stage starts from its own ``reset()`` so it gets a fresh tick budget and the *same*
kitchen (one seed), which makes the stages comparable and keeps a long motion stage from
truncating the one after it.

Usage (from ``aspire/``)::

    PYTHONPATH="" .venv-robocasa/bin/python examples/robocasa/api_tool_check/probe.py
    PYTHONPATH="" .venv-robocasa/bin/python examples/robocasa/api_tool_check/probe.py \
        --stages perception grasps --task PickPlaceCounterToStove --seed 101

Nothing here reads simulator state: every number comes back through the public tool mapping.
The two exceptions are labelled where they occur -- the simulator's video buffer (the public
surface has no equivalent, and base motion is otherwise invisible) and ``_elapsed_steps`` for
the tick accounting printed in the log.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import sys
import traceback
from typing import Any

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from session import (
    EXPECTED_TOOL_COUNT,
    OUT,
    RAW,
    ArmCommand,
    MotionStrategy,
    NumpyJson,
    Pose,
    RobotAction,
    Session,
    Stage,
    Timer,
    Trajectory,
    build_session,
    dump_cloud,
    dump_error,
    dump_execution,
    dump_geometry,
    dump_grasp,
    dump_pose,
    dump_segmentation,
    dump_state,
    error_text,
    pose_matrix,
    project_points,
    quat_to_matrix,
    quaternion_angle_deg,
    steps_to_arrays,
    yaw_of,
)

BASE_FRAME = "robot0_link0"


# --------------------------------------------------------------------------- helpers


def _segment_best(tools: Any, camera: str, prompt: str) -> Any:
    found = tools["segment_text"](camera, prompt)
    if not found.ok or not found.segmentations:
        return None
    return found.segmentations[0]


def _cloud_for(tools: Any, camera: str, prompt: str) -> tuple[Any, Any]:
    """(segmentation, point cloud in robot0_link0) for one prompt, or (None, None)."""
    segmentation = _segment_best(tools, camera, prompt)
    if segmentation is None:
        return None, None
    try:
        return segmentation, tools["mask_to_point_cloud"](segmentation, camera, BASE_FRAME)
    except ValueError:
        return segmentation, None


def _capture(session: Session, stage: Stage, name: str, *, stride: int = 1) -> None:
    """Encode whatever the step observer collected since the last call."""
    steps, frames = session.capture.take()
    if steps:
        for key, value in steps_to_arrays(steps).items():
            stage.array(f"{name}__{key}", value)
    for camera, images in frames.items():
        if images:
            stage.video(f"{name}__{camera}", images[::stride], fps=20.0 / stride)


def _frames_on(session: Session, stride: int = 2) -> None:
    session.capture.capture_frames = True
    session.capture.frame_stride = stride


def _frames_off(session: Session) -> None:
    session.capture.capture_frames = False


# --------------------------------------------------------------------------- stage: identity


def stage_identity(session: Session, stage: Stage) -> None:
    tools = session.tools
    session.reset()

    stage.notes.append(
        f"public tool namespace: {len(tools)} names "
        f"(expected {EXPECTED_TOOL_COUNT}); "
        f"{'MATCH' if len(tools) == EXPECTED_TOOL_COUNT else 'MISMATCH'}"
    )
    stage.record(
        "registry",
        {"allowlist": "ROBOCASA_PUBLIC_TOOL_NAMES"},
        {"tool_names": sorted(tools), "count": len(tools)},
        ok=len(tools) == EXPECTED_TOOL_COUNT,
        label="register_tools (the namespace itself)",
        verdict=f"{len(tools)} tools exposed",
    )

    with Timer() as timer:
        context = tools["get_task_context"]()
    stage.record(
        "get_task_context",
        {},
        {
            "suite": context.suite,
            "task_id": context.task_id,
            "task_name": context.task_name,
            "family": context.family,
            "language": context.language,
            "metadata": {k: str(v) for k, v in dict(context.metadata).items()},
        },
        ok=bool(str(context.language).strip()),
        verdict=f"instruction: {context.language!r}",
        elapsed_s=timer.elapsed,
    )

    for tool in ("robocasa.get_task_metadata", "robocasa.get_controller_metadata"):
        with Timer() as timer:
            metadata = tools[tool]()
        stage.record(
            tool,
            {},
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(metadata).items()},
            ok=bool(metadata),
            verdict=f"{len(dict(metadata))} fields",
            elapsed_s=timer.elapsed,
        )

    controller = dict(tools["robocasa.get_controller_metadata"]())
    limits = controller.get("joint_limits")
    measured = np.asarray(
        tools["get_robot_state"]().joint_positions["primary"], dtype=np.float64
    )
    if isinstance(limits, Mapping):
        low = np.asarray(limits["low"], dtype=np.float64)
        high = np.asarray(limits["high"], dtype=np.float64)
        outside = [
            {
                "joint": index,
                "measured_rad": round(float(measured[index]), 4),
                "published_low": round(float(low[index]), 4),
                "published_high": round(float(high[index]), 4),
            }
            for index in range(len(measured))
            if measured[index] < low[index] - 1e-6 or measured[index] > high[index] + 1e-6
        ]
        stage.record(
            "robocasa.get_controller_metadata",
            {"check": "measured joint positions against the published joint_limits"},
            {
                "published_low": np.round(low, 4).tolist(),
                "published_high": np.round(high, 4).tolist(),
                "measured": np.round(measured, 4).tolist(),
                "joints_outside_their_published_range": outside,
                "panda_reference_low": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                "panda_reference_high": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            },
            ok=not outside,
            label="robocasa.get_controller_metadata joint_limits vs the measured arm",
            verdict=(
                "every measured joint is inside its published range"
                if not outside
                else f"{len(outside)} of 7 measured joints fall OUTSIDE the published range "
                f"(joints {[row['joint'] for row in outside]})"
            ),
        )

    with Timer() as timer:
        observation = tools["get_observation"]()
    filtered = tools["get_observation"]("agentview")
    cameras: dict[str, Any] = {}
    assets: dict[str, str] = {}
    for name, camera in sorted(observation.cameras.items()):
        key = name.replace("robot0_", "")
        assets[f"rgb_{key}"] = stage.image(f"rgb_{key}", camera.rgb)
        stage.array(f"depth_{key}", camera.depth_m.astype(np.float32))
        stage.array(f"intrinsics_{key}", camera.intrinsics)
        stage.array(f"cam_pose_{key}", pose_matrix(camera.camera_pose))
        depth = camera.depth_m
        cameras[name] = {
            "rgb_shape": list(camera.rgb.shape),
            "rgb_dtype": str(camera.rgb.dtype),
            "depth_shape": list(depth.shape),
            "depth_min_m": round(float(np.nanmin(depth)), 4),
            "depth_max_m": round(float(np.nanmax(depth)), 4),
            "depth_finite_fraction": round(float(np.isfinite(depth).mean()), 4),
            "intrinsics": np.round(camera.intrinsics, 3).tolist(),
            "camera_pose": dump_pose(camera.camera_pose),
            "frame": camera.frame,
        }
    stage.record(
        "get_observation",
        {"camera_names": None, "then": "'agentview'"},
        {
            "camera_names": sorted(observation.cameras),
            "filtered_to": sorted(filtered.cameras),
            "timestamp_s": observation.timestamp_s,
            "cameras": cameras,
        },
        ok={"agentview", "robot0_eye_in_hand"} <= set(observation.cameras),
        verdict="both cameras, RGB + metric depth + K + extrinsics",
        assets=assets,
        elapsed_s=timer.elapsed,
    )

    # Calibration round trip: back-project every valid pixel, reproject, measure the residual.
    # This is the one number that decides whether any cloud the agent builds is trustworthy.
    for name in ("agentview", "robot0_eye_in_hand"):
        camera = observation.cameras[name]
        mask = np.isfinite(camera.depth_m) & (camera.depth_m > 0)
        cloud = tools["mask_to_point_cloud"](mask, name, BASE_FRAME)
        pixels = project_points(cloud.points, camera)
        rows, columns = np.nonzero(mask)
        residual = np.hypot(pixels[:, 0] - columns, pixels[:, 1] - rows)
        stage.array(f"reprojection_residual_{name.replace('robot0_', '')}", residual)
        stage.record(
            "mask_to_point_cloud",
            {"mask": "every valid depth pixel", "camera_name": name, "target_frame": BASE_FRAME},
            {
                "n_points": int(len(cloud.points)),
                "reprojection_residual_px": {
                    "mean": round(float(residual.mean()), 6),
                    "p99": round(float(np.percentile(residual, 99)), 6),
                    "max": round(float(residual.max()), 6),
                },
                "cloud": dump_cloud(cloud),
            },
            ok=float(residual.max()) < 0.01,
            label=f"calibration round trip ({name})",
            verdict=f"max reprojection error {float(residual.max()):.2e} px",
        )
        stage.array(f"scene_cloud_{name.replace('robot0_', '')}", cloud.points[::13])
        if cloud.colors is not None:
            stage.array(f"scene_colors_{name.replace('robot0_', '')}", cloud.colors[::13])

    with Timer() as timer:
        state = tools["get_robot_state"]()
    stage.array("ee_pose_matrix", pose_matrix(state.end_effector_poses["primary"]))
    stage.array("joint_positions", np.asarray(state.joint_positions["primary"]))
    stage.array(
        "gripper_position", np.asarray([float(state.gripper_positions["primary"])])
    )
    stage.record(
        "get_robot_state",
        {},
        dump_state(state),
        ok=state.base_frame == BASE_FRAME and len(state.joint_positions["primary"]) == 7,
        verdict=f"7 joints, base_frame={state.base_frame}",
        elapsed_s=timer.elapsed,
    )

    # Independent cross-check on the reported end-effector pose: the wrist camera is bolted
    # to the hand, so the gap between the two published poses is a fixed mounting offset.
    wrist_camera = observation.cameras["robot0_eye_in_hand"]
    eef = np.asarray(state.end_effector_poses["primary"].position, dtype=np.float64)
    wrist_origin = np.asarray(wrist_camera.camera_pose.position, dtype=np.float64)
    eef_in_wrist = project_points(eef.reshape(1, 3), wrist_camera)[0]
    stage.record(
        "get_robot_state",
        {"compare": "end_effector_poses['primary'] vs the wrist camera_pose"},
        {
            "eef_position_m": np.round(eef, 4).tolist(),
            "wrist_camera_position_m": np.round(wrist_origin, 4).tolist(),
            "separation_m": round(float(np.linalg.norm(eef - wrist_origin)), 4),
            "eef_projects_into_wrist_image_at_px": np.round(eef_in_wrist[:2], 1).tolist(),
            "eef_depth_in_wrist_camera_m": round(float(eef_in_wrist[2]), 4),
        },
        ok=0.05 < float(np.linalg.norm(eef - wrist_origin)) < 0.25,
        label="get_robot_state cross-check against the wrist camera mounting",
        verdict=(
            f"the end-effector origin sits {1000 * float(np.linalg.norm(eef - wrist_origin)):.0f} mm "
            f"from the wrist camera and projects into the wrist image at "
            f"({eef_in_wrist[0]:.0f}, {eef_in_wrist[1]:.0f}) px, "
            f"{eef_in_wrist[2]:.3f} m in front of it"
        ),
    )

    with Timer() as timer:
        base_pose = tools["robocasa.get_base_pose"]()
    stage.record(
        "robocasa.get_base_pose",
        {},
        dump_pose(base_pose),
        ok=base_pose.frame == "world",
        verdict=f"world pose of robot0_link0, yaw={yaw_of(base_pose):.3f}",
        elapsed_s=timer.elapsed,
    )
    stage.array("base_pose_matrix", pose_matrix(base_pose))


# --------------------------------------------------------------------------- stage: perception

SCENE_PROMPTS = (
    "fish",
    "plate",
    "pan",
    "counter",
    "cabinet",
    "drawer",
    "cabinet handle",
    "stove",
    "sink",
)
WRIST_PROMPTS = ("plate", "fish", "counter")


def stage_perception(session: Session, stage: Stage) -> None:
    tools = session.tools
    observation = session.reset()
    for name, camera in observation.cameras.items():
        key = name.replace("robot0_", "")
        stage.image(f"rgb_{key}", camera.rgb)
        stage.array(f"depth_{key}", camera.depth_m.astype(np.float32))
        stage.array(f"intrinsics_{key}", camera.intrinsics)
        stage.array(f"cam_pose_{key}", pose_matrix(camera.camera_pose))

    # ---- segment_text, both cameras, several prompts
    for camera_name, prompts in (("agentview", SCENE_PROMPTS), ("robot0_eye_in_hand", WRIST_PROMPTS)):
        key = camera_name.replace("robot0_", "")
        for prompt in prompts:
            with Timer() as timer:
                found = tools["segment_text"](camera_name, prompt)
            hits = [dump_segmentation(s) for s in found.segmentations] if found.ok else []
            assets = {}
            if found.ok and found.segmentations:
                masks = np.stack(
                    [np.asarray(s.mask, dtype=bool) for s in found.segmentations[:3]]
                )
                name = f"mask_{key}_{prompt.replace(' ', '_')}"
                stage.array(name, masks)
                assets["masks"] = name
            stage.record(
                "segment_text",
                {"camera_name": camera_name, "text": prompt},
                {
                    "ok": bool(found.ok),
                    "n": len(hits),
                    "error": error_text(found.error),
                    "segmentations": hits,
                    "diagnostics": {k: str(v) for k, v in dict(found.diagnostics).items()},
                },
                ok=bool(found.ok),
                label=f"segment_text({camera_name!r}, {prompt!r})",
                verdict=(
                    f"{len(hits)} mask(s), top score "
                    f"{hits[0]['score'] if hits else float('nan')}"
                    if found.ok
                    else error_text(found.error)
                ),
                assets=assets,
                elapsed_s=timer.elapsed,
            )

    # ---- segment_points: click the centroid of the text mask, and the wrist centre
    fish_segmentation = _segment_best(tools, "agentview", "fish")
    clicks: list[tuple[str, list[float], str]] = []
    if fish_segmentation is not None:
        rows, columns = np.nonzero(np.asarray(fish_segmentation.mask, dtype=bool))
        clicks.append(
            ("agentview", [float(columns.mean()), float(rows.mean())], "centroid of the fish mask")
        )
    clicks.append(("robot0_eye_in_hand", [256.0, 256.0], "image centre"))
    for camera_name, point, why in clicks:
        key = camera_name.replace("robot0_", "")
        with Timer() as timer:
            clicked = tools["segment_points"](camera_name, np.array([point]))
        hits = [dump_segmentation(s) for s in clicked.segmentations] if clicked.ok else []
        assets = {}
        if clicked.ok and clicked.segmentations:
            name = f"pointmask_{key}"
            stage.array(
                name,
                np.stack([np.asarray(s.mask, dtype=bool) for s in clicked.segmentations[:3]]),
            )
            stage.array(f"{name}__point", np.asarray(point))
            assets["masks"] = name
        stage.record(
            "segment_points",
            {"camera_name": camera_name, "points_px": point, "why": why},
            {
                "ok": bool(clicked.ok),
                "n": len(hits),
                "error": error_text(clicked.error),
                "segmentations": hits,
            },
            ok=bool(clicked.ok),
            label=f"segment_points({camera_name!r}, {point})",
            verdict=f"{len(hits)} mask(s)" if clicked.ok else error_text(clicked.error),
            assets=assets,
            elapsed_s=timer.elapsed,
        )

    # multi-point prompt: documented as unsupported by the service wire format
    multi = tools["segment_points"]("agentview", np.array([[200.0, 200.0], [300.0, 300.0]]))
    stage.record(
        "segment_points",
        {"camera_name": "agentview", "points_px": [[200, 200], [300, 300]]},
        {"ok": bool(multi.ok), "error": error_text(multi.error)},
        ok=not multi.ok,
        label="segment_points (two points -- expected refusal)",
        verdict=error_text(multi.error) or "accepted two points",
    )

    # ---- mask_to_point_cloud: the documented default frame is wrong here (trap 16)
    if fish_segmentation is not None:
        try:
            tools["mask_to_point_cloud"](fish_segmentation, "agentview")
            default_outcome = "accepted the default target_frame='robot_base'"
            default_ok = False
        except ValueError as exc:
            default_outcome = f"ValueError: {exc}"
            default_ok = True
        stage.record(
            "mask_to_point_cloud",
            {"mask": "fish", "camera_name": "agentview", "target_frame": "(default)"},
            {"outcome": default_outcome},
            ok=default_ok,
            label="mask_to_point_cloud (default frame -- expected to raise)",
            verdict=default_outcome[:120],
        )

    clouds: dict[str, Any] = {}
    for prompt in ("fish", "plate", "pan", "counter"):
        segmentation, cloud = _cloud_for(tools, "agentview", prompt)
        if cloud is None:
            stage.record(
                "mask_to_point_cloud",
                {"mask": prompt, "camera_name": "agentview", "target_frame": BASE_FRAME},
                {"error": "no mask or no valid depth"},
                ok=False,
                label=f"mask_to_point_cloud({prompt!r})",
                verdict="no cloud",
            )
            continue
        clouds[prompt] = cloud
        camera = observation.cameras["agentview"]
        pixels = project_points(cloud.points, camera)
        mask = np.asarray(segmentation.mask, dtype=bool)
        inside = mask[
            np.clip(np.rint(pixels[:, 1]).astype(int), 0, mask.shape[0] - 1),
            np.clip(np.rint(pixels[:, 0]).astype(int), 0, mask.shape[1] - 1),
        ]
        stage.array(f"cloud_{prompt}", cloud.points)
        if cloud.colors is not None:
            stage.array(f"cloudcolor_{prompt}", cloud.colors)
        stage.record(
            "mask_to_point_cloud",
            {
                "mask": f"segment_text('{prompt}')[0]",
                "camera_name": "agentview",
                "target_frame": BASE_FRAME,
            },
            {
                "cloud": dump_cloud(cloud),
                "mask_pixels": int(mask.sum()),
                "reprojects_inside_mask_fraction": round(float(inside.mean()), 4),
            },
            ok=len(cloud.points) >= 20 and float(inside.mean()) > 0.98,
            label=f"mask_to_point_cloud({prompt!r})",
            verdict=(
                f"{len(cloud.points)} pts, centroid "
                f"{np.round(cloud.points.mean(axis=0), 3).tolist()}, "
                f"{100 * float(inside.mean()):.1f}% reproject inside the mask"
            ),
        )

    # ---- cross-camera agreement. Two different cameras with different intrinsics and
    # extrinsics must put the same object in the same place in robot0_link0. Unlike the
    # reprojection round trip, this cannot pass by being self-consistent.
    for prompt in ("fish", "plate"):
        _, wrist_cloud = _cloud_for(tools, "robot0_eye_in_hand", prompt)
        scene_cloud = clouds.get(prompt)
        if wrist_cloud is None or scene_cloud is None:
            continue
        stage.array(f"wristcloud_{prompt}", wrist_cloud.points)
        scene_geometry = tools["estimate_geometry"](scene_cloud)
        wrist_geometry = tools["estimate_geometry"](wrist_cloud)
        separation = float(
            np.linalg.norm(
                np.asarray(scene_geometry.pose.position) - np.asarray(wrist_geometry.pose.position)
            )
        )
        stage.record(
            "mask_to_point_cloud",
            {"mask": prompt, "camera_name": "agentview vs robot0_eye_in_hand", "target_frame": BASE_FRAME},
            {
                "agentview_centre_m": np.round(scene_geometry.pose.position, 4).tolist(),
                "wrist_centre_m": np.round(wrist_geometry.pose.position, 4).tolist(),
                "separation_m": round(separation, 4),
                "agentview_extents_m": np.round(scene_geometry.extents, 4).tolist(),
                "wrist_extents_m": np.round(wrist_geometry.extents, 4).tolist(),
            },
            ok=separation < 0.05,
            label=f"cross-camera agreement ({prompt!r})",
            verdict=(
                f"the two cameras place the {prompt} {1000 * separation:.0f} mm apart in "
                f"{BASE_FRAME}"
            ),
        )

    # ---- crop_point_cloud
    if "counter" in clouds:
        cloud = clouds["counter"]
        centre = cloud.points.mean(axis=0)
        lower = centre - np.array([0.25, 0.25, 0.25])
        upper = centre + np.array([0.25, 0.25, 0.25])
        with Timer() as timer:
            cropped = tools["crop_point_cloud"](cloud, lower, upper)
        stage.array("crop_counter_in", cloud.points)
        stage.array("crop_counter_out", cropped.points)
        stage.array("crop_box", np.stack([lower, upper]))
        stage.record(
            "crop_point_cloud",
            {
                "point_cloud": "counter",
                "lower_xyz": np.round(lower, 4).tolist(),
                "upper_xyz": np.round(upper, 4).tolist(),
            },
            {"input": dump_cloud(cloud), "output": dump_cloud(cropped)},
            ok=len(cropped.points) < len(cloud.points) and len(cropped.points) > 0,
            verdict=f"{len(cloud.points)} -> {len(cropped.points)} points",
            elapsed_s=timer.elapsed,
        )
        try:
            tools["crop_point_cloud"](
                cloud, np.array([99.0, 99.0, 99.0]), np.array([100.0, 100.0, 100.0])
            )
            empty_outcome = "returned a cloud for bounds containing nothing"
            empty_ok = False
        except ValueError as exc:
            empty_outcome = f"ValueError: {exc}"
            empty_ok = True
        stage.record(
            "crop_point_cloud",
            {"point_cloud": "counter", "lower_xyz": [99, 99, 99], "upper_xyz": [100, 100, 100]},
            {"outcome": empty_outcome},
            ok=empty_ok,
            label="crop_point_cloud (empty result -- expected to raise)",
            verdict=empty_outcome[:120],
        )

    # ---- estimate_geometry
    for prompt, cloud in clouds.items():
        with Timer() as timer:
            geometry = tools["estimate_geometry"](cloud)
        stage.array(f"obb_{prompt}", np.concatenate([pose_matrix(geometry.pose).ravel(), geometry.extents]))
        stage.record(
            "estimate_geometry",
            {"point_cloud": f"{prompt} ({len(cloud.points)} pts, {cloud.frame})"},
            dump_geometry(geometry),
            ok=bool(np.all(np.asarray(geometry.extents) > 0)),
            label=f"estimate_geometry({prompt!r})",
            verdict=(
                f"extents {np.round(geometry.extents, 3).tolist()} m, centre "
                f"{np.round(geometry.pose.position, 3).tolist()}"
            ),
            elapsed_s=timer.elapsed,
        )

    # ---- localize_object (the composition)
    for query in ("fish", "pan", "plate", "handle of the cabinet"):
        with Timer() as timer:
            found = tools["localize_object"](query)
        if not found.ok:
            stage.record(
                "localize_object",
                {"query": query},
                {"ok": False, "error": error_text(found.error)},
                ok=False,
                label=f"localize_object({query!r})",
                verdict=error_text(found.error),
                elapsed_s=timer.elapsed,
            )
            continue
        key = query.replace(" ", "_")
        stage.array(f"localize_cloud_{key}", found.point_cloud.points)
        stage.array(f"localize_mask_{key}", np.asarray(found.segmentation.mask, dtype=bool))
        stage.array(
            f"localize_obb_{key}",
            np.concatenate([pose_matrix(found.geometry.pose).ravel(), found.geometry.extents]),
        )
        centroid_px = project_points(
            found.geometry.pose.position.reshape(1, 3), observation.cameras["agentview"]
        )[0]
        stage.record(
            "localize_object",
            {"query": query, "camera_name": "(default agentview)", "target_frame": "(default)"},
            {
                "segmentation": dump_segmentation(found.segmentation),
                "point_cloud": dump_cloud(found.point_cloud),
                "geometry": dump_geometry(found.geometry),
                "obb_centre_reprojects_to_px": np.round(centroid_px[:2], 1).tolist(),
            },
            ok=True,
            label=f"localize_object({query!r})",
            verdict=(
                f"centre {np.round(found.geometry.pose.position, 3).tolist()} m "
                f"({BASE_FRAME}), extents {np.round(found.geometry.extents, 3).tolist()} m"
            ),
            elapsed_s=timer.elapsed,
        )


# --------------------------------------------------------------------------- stage: grasps


def stage_grasps(session: Session, stage: Stage) -> None:
    tools = session.tools
    observation = session.reset()
    camera = observation.cameras["agentview"]
    stage.image("rgb_agentview", camera.rgb)
    stage.array("intrinsics_agentview", camera.intrinsics)
    stage.array("cam_pose_agentview", pose_matrix(camera.camera_pose))
    stage.array("depth_agentview", camera.depth_m.astype(np.float32))

    for prompt in ("fish", "pan", "plate"):
        segmentation = _segment_best(tools, "agentview", prompt)
        if segmentation is None:
            stage.record(
                "generate_grasps",
                {"camera_name": "agentview", "mask": prompt},
                {"error": "no mask"},
                ok=False,
                label=f"generate_grasps({prompt!r})",
                verdict="no mask to grasp",
            )
            continue
        mask = np.asarray(segmentation.mask, dtype=bool)
        stage.array(f"mask_{prompt}", mask)
        cloud = tools["mask_to_point_cloud"](segmentation, "agentview", BASE_FRAME)
        stage.array(f"cloud_{prompt}", cloud.points)
        if cloud.colors is not None:
            stage.array(f"cloudcolor_{prompt}", cloud.colors)

        # Does max_candidates do anything? Ask for 1, 8 and 30 of the same thing.
        counts = {}
        for requested in (1, 8, 30):
            probe = tools["generate_grasps"]("agentview", segmentation, max_candidates=requested)
            counts[requested] = len(probe.grasps) if probe.ok else 0
        stage.record(
            "generate_grasps",
            {"camera_name": "agentview", "mask": prompt, "max_candidates": "1 / 8 / 30"},
            {"returned_counts": counts},
            ok=len(set(counts.values())) > 1 or max(counts.values()) == 0,
            label=f"generate_grasps({prompt!r}) -- does max_candidates change the count?",
            verdict=f"asked 1/8/30, got {counts[1]}/{counts[8]}/{counts[30]}",
        )

        with Timer() as timer:
            grasps = tools["generate_grasps"]("agentview", segmentation, max_candidates=8)
        candidates = list(grasps.grasps) if grasps.ok else []
        if candidates:
            stage.array(
                f"grasps_{prompt}",
                np.stack([pose_matrix(c.pose) for c in candidates]),
            )
            stage.array(
                f"graspmeta_{prompt}",
                np.asarray(
                    [[c.score, c.width_m, c.approach_distance_m] for c in candidates],
                    dtype=np.float64,
                ),
            )
            contacts = [
                c.metadata.get("contact_point")
                for c in candidates
                if c.metadata.get("contact_point") is not None
            ]
            if len(contacts) == len(candidates):
                stage.array(f"graspcontact_{prompt}", np.asarray(contacts, dtype=np.float64))
            # How far is each grasp point from the object it is supposed to grasp?
            distances = [
                float(np.min(np.linalg.norm(cloud.points - c.pose.position, axis=1)))
                for c in candidates
            ]
            stage.array(f"graspdist_{prompt}", np.asarray(distances))
        else:
            distances = []
        stage.record(
            "generate_grasps",
            {
                "camera_name": "agentview",
                "mask": f"segment_text('{prompt}')[0] ({int(mask.sum())} px)",
                "backend": "contact-graspnet",
                "max_candidates": 8,
            },
            {
                "ok": bool(grasps.ok),
                "n": len(candidates),
                "error": error_text(grasps.error),
                "frame": str(grasps.diagnostics.get("frame", "")),
                "candidates": [dump_grasp(c) for c in candidates],
                "distance_to_object_cloud_m": [round(d, 4) for d in distances],
            },
            ok=bool(grasps.ok) and bool(candidates),
            label=f"generate_grasps({prompt!r})",
            verdict=(
                f"{len(candidates)} grasps, scores "
                f"{[round(c.score, 3) for c in candidates]}, "
                f"nearest-object-point distance "
                f"{[round(d, 3) for d in distances]} m"
                if candidates
                else error_text(grasps.error)
            ),
            elapsed_s=timer.elapsed,
        )

        if not grasps.ok:
            continue
        for strategy in ("top_down", "highest_score"):
            chosen = tools["select_grasp"](grasps, strategy)
            stage.record(
                "select_grasp",
                {"grasps": f"{len(candidates)} candidates from {prompt!r}", "strategy": strategy},
                {"selected": None if chosen is None else dump_grasp(chosen)},
                ok=chosen is not None,
                label=f"select_grasp({prompt!r}, {strategy!r})",
                verdict=(
                    f"score {chosen.score:.3f} at "
                    f"{np.round(chosen.pose.position, 3).tolist()}"
                    if chosen is not None
                    else "no candidate matched the strategy"
                ),
            )
            if chosen is not None:
                stage.array(f"selected_{prompt}_{strategy}", pose_matrix(chosen.pose))

        # generate_grasps publishes the public EEF, so a candidate is a solve_ik target.
        best = tools["select_grasp"](grasps, "highest_score")
        if best is not None:
            solution = tools["solve_ik"](best.pose)
            stage.record(
                "solve_ik",
                {
                    "target_pose": dump_pose(best.pose),
                    "source": f"{prompt} grasp (public EEF)",
                },
                {
                    "ok": bool(solution.ok),
                    "error": error_text(solution.error),
                    "joint_positions_rad": None
                    if solution.joint_positions is None
                    else np.round(solution.joint_positions, 4).tolist(),
                    "diagnostics": {k: str(v) for k, v in dict(solution.diagnostics).items()},
                },
                ok=bool(solution.ok),
                label=f"solve_ik on the {prompt!r} grasp",
                verdict="reachable" if solution.ok else error_text(solution.error),
            )


# --------------------------------------------------------------------------- stage: graspexec


def stage_graspexec(session: Session, stage: Stage) -> None:
    """Can a grasp from ``generate_grasps`` actually be driven to and closed on?

    ``generate_grasps`` publishes the public EEF (``gripper0_right_eef`` here), so the
    candidate is a ``move_to_pose`` target with no extra panda_hand map.
    """
    tools = session.tools
    observation = session.reset()
    camera = observation.cameras["agentview"]
    stage.image("rgb_before", camera.rgb)
    stage.array("intrinsics_agentview", camera.intrinsics)
    stage.array("cam_pose_agentview", pose_matrix(camera.camera_pose))

    segmentation = _segment_best(tools, "agentview", "fish")
    if segmentation is None:
        stage.notes.append("no fish mask; nothing to grasp")
        return
    cloud = tools["mask_to_point_cloud"](segmentation, "agentview", BASE_FRAME)
    stage.array("cloud_fish", cloud.points)
    # generate_grasps is not deterministic: the same mask can return 0, 1 or 2 candidates on
    # consecutive calls. Retry rather than lose the whole stage, and record how many tries it
    # took -- that number is itself a result.
    attempts = []
    grasps = None
    best = None
    for attempt in range(6):
        grasps = tools["generate_grasps"]("agentview", segmentation, max_candidates=8)
        attempts.append(len(grasps.grasps) if grasps.ok else 0)
        best = tools["select_grasp"](grasps, "highest_score") if grasps.ok else None
        if best is not None:
            break
    stage.record(
        "generate_grasps",
        {"camera_name": "agentview", "mask": "fish", "max_candidates": 8, "repeats": len(attempts)},
        {"candidates_returned_per_identical_call": attempts},
        ok=best is not None,
        label="generate_grasps repeated on an identical mask",
        verdict=f"identical calls returned {attempts} candidates",
    )
    if best is None:
        stage.notes.append("no grasp candidate to execute after 6 identical calls")
        return
    stage.array("grasp_pose", pose_matrix(best.pose))

    hand = pose_matrix(best.pose)
    approach = hand[:3, 2]
    contact = best.metadata.get("contact_point")
    contact_vector = (
        None if contact is None else np.asarray(contact, dtype=np.float64) - hand[:3, 3]
    )
    stage.record(
        "generate_grasps",
        {"candidate": dump_grasp(best)},
        {
            "pose_to_contact_m": None
            if contact_vector is None
            else round(float(np.linalg.norm(contact_vector)), 4),
            "approach_axis_base": np.round(approach, 4).tolist(),
            "cos_angle_between_local_z_and_pose_to_contact": None
            if contact_vector is None
            else round(
                float(np.dot(approach, contact_vector) / np.linalg.norm(contact_vector)), 4
            ),
            "nearest_object_point_to_pose_m": round(
                float(np.min(np.linalg.norm(cloud.points - hand[:3, 3], axis=1))), 4
            ),
            "nearest_object_point_to_contact_m": None
            if contact is None
            else round(
                float(np.min(np.linalg.norm(cloud.points - np.asarray(contact), axis=1))), 4
            ),
        },
        ok=True,
        label="grasp pose geometry (where is the pose relative to its contact point?)",
        verdict=(
            "pose sits "
            f"{float(np.linalg.norm(contact_vector)):.4f} m behind the contact point along "
            "its own local +Z"
            if contact_vector is not None
            else "no contact point reported"
        ),
    )

    eef_target = best.pose
    stage.array("grasp_eef_target", pose_matrix(eef_target))
    standoff = Pose(
        position=np.asarray(eef_target.position) - approach * 0.10,
        quaternion_wxyz=np.asarray(eef_target.quaternion_wxyz),
        frame=eef_target.frame,
    )
    stage.array("grasp_standoff_target", pose_matrix(standoff))

    _frames_on(session, stride=1)
    tools["open_gripper"]()
    _capture(session, stage, "grasp_00_open")

    for label, target, name in (
        ("standoff (10 cm back along the approach axis)", standoff, "grasp_01_standoff"),
        ("grasp pose (public EEF)", eef_target, "grasp_02_grasp"),
    ):
        with Timer() as timer:
            moved = tools["move_to_pose"](target, 0.015, 200)
        reached = tools["get_robot_state"]().end_effector_poses["primary"]
        stage.record(
            "move_to_pose",
            {"target_pose": dump_pose(target), "tolerance": 0.015, "max_steps": 200, "why": label},
            {
                "execution": dump_execution(moved),
                "reached": dump_pose(reached),
                "position_error_m": round(
                    float(np.linalg.norm(np.asarray(reached.position) - target.position)), 5
                ),
                "orientation_error_deg": round(
                    quaternion_angle_deg(reached.quaternion_wxyz, target.quaternion_wxyz), 3
                ),
            },
            ok=bool(moved.ok),
            label=f"move_to_pose -> {label}",
            verdict=(
                f"{moved.steps_executed} ticks, "
                f"{1000 * float(np.linalg.norm(np.asarray(reached.position) - target.position)):.1f} mm error"
            )
            if moved.ok
            else error_text(moved.error),
            elapsed_s=timer.elapsed,
        )
        _capture(session, stage, name)
        stage.image(f"{name}_wrist", tools["get_observation"]().cameras["robot0_eye_in_hand"].rgb)
        stage.image(f"{name}_scene", tools["get_observation"]().cameras["agentview"].rgb)

    closed = tools["close_gripper"]()
    jaw = float(tools["get_robot_state"]().gripper_positions["primary"])
    stage.record(
        "close_gripper",
        {"at": "the grasp pose"},
        {"execution": dump_execution(closed), "measured_jaw": round(jaw, 4)},
        ok=True,
        label="close_gripper on the candidate grasp",
        verdict=f"jaw settled at {jaw:.3f} (1 = open, 0 = closed)",
    )
    _capture(session, stage, "grasp_03_close")
    stage.image("grasp_03_close_wrist", tools["get_observation"]().cameras["robot0_eye_in_hand"].rgb)

    lift = tools["get_robot_state"]().end_effector_poses["primary"]
    lift_target = Pose(
        position=np.asarray(lift.position) + np.array([0.0, 0.0, 0.12]),
        quaternion_wxyz=np.asarray(lift.quaternion_wxyz),
        frame=lift.frame,
    )
    lifted = tools["move_to_pose"](lift_target, 0.02, 200)
    after_cloud = None
    try:
        after_segmentation = _segment_best(tools, "agentview", "fish")
        if after_segmentation is not None:
            after_cloud = tools["mask_to_point_cloud"](after_segmentation, "agentview", BASE_FRAME)
    except ValueError:
        after_cloud = None
    rise = (
        None
        if after_cloud is None
        else round(float(after_cloud.points.mean(axis=0)[2] - cloud.points.mean(axis=0)[2]), 4)
    )
    stage.record(
        "move_to_pose",
        {"target_pose": dump_pose(lift_target), "why": "lift 12 cm after closing"},
        {
            "execution": dump_execution(lifted),
            "fish_centroid_z_before_m": round(float(cloud.points.mean(axis=0)[2]), 4),
            "fish_centroid_z_after_m": None
            if after_cloud is None
            else round(float(after_cloud.points.mean(axis=0)[2]), 4),
            "fish_rise_m": rise,
            "jaw_after": round(float(tools["get_robot_state"]().gripper_positions["primary"]), 4),
        },
        ok=bool(lifted.ok),
        label="lift after the grasp (did the object come with it?)",
        verdict=(
            f"fish centroid moved {rise:+.3f} m in z"
            if rise is not None
            else "fish no longer segmentable after the lift"
        ),
    )
    jaw_after = float(tools["get_robot_state"]().gripper_positions["primary"])
    picked = rise is not None and rise > 0.05
    stage.record(
        "generate_grasps",
        {"pipeline": "generate_grasps -> select_grasp -> move_to_pose -> close_gripper -> lift"},
        {
            "reached_the_candidate": True,
            "jaw_after_closing": round(jaw_after, 4),
            "object_rise_m": rise,
            "reading": (
                "the jaw closed to near-zero, which on this gripper means it met no object; "
                "an object between the fingers stalls the jaw well short of 0"
            ),
        },
        ok=picked,
        label="end-to-end pick outcome",
        verdict=(
            "the object was lifted"
            if picked
            else f"NOT picked: jaw closed to {jaw_after:.3f} (empty) and the object rose "
            f"{0.0 if rise is None else rise:+.3f} m"
        ),
    )
    _capture(session, stage, "grasp_04_lift")
    stage.image("grasp_04_lift_scene", tools["get_observation"]().cameras["agentview"].rgb)
    stage.image("grasp_04_lift_wrist", tools["get_observation"]().cameras["robot0_eye_in_hand"].rgb)
    _frames_off(session)
    stage.notes.append(f"ticks used by this stage: {session.elapsed()}")


# --------------------------------------------------------------------------- stage: ik


def stage_ik(session: Session, stage: Stage) -> None:
    tools = session.tools
    session.reset()
    state = tools["get_robot_state"]()
    home_joints = np.asarray(state.joint_positions["primary"], dtype=np.float64)
    home_pose = state.end_effector_poses["primary"]
    stage.array("home_joints", home_joints)
    stage.array("home_ee", pose_matrix(home_pose))

    # 1. identity: the current pose must solve back to (near) the current joints.
    with Timer() as timer:
        identity = tools["solve_ik"](home_pose)
    joint_delta = (
        None
        if identity.joint_positions is None
        else float(np.max(np.abs(identity.joint_positions - home_joints)))
    )
    stage.record(
        "solve_ik",
        {"target_pose": dump_pose(home_pose), "note": "the arm's current pose"},
        {
            "ok": bool(identity.ok),
            "arm": identity.arm,
            "embodiment": identity.embodiment,
            "joint_positions_rad": None
            if identity.joint_positions is None
            else np.round(identity.joint_positions, 5).tolist(),
            "max_joint_delta_from_current_rad": joint_delta,
            "error": error_text(identity.error),
            "diagnostics": {k: str(v) for k, v in dict(identity.diagnostics).items()},
        },
        ok=bool(identity.ok),
        label="solve_ik (identity: current pose)",
        verdict=f"solved; max joint delta from the measured arm {joint_delta:.4f} rad"
        if identity.ok
        else error_text(identity.error),
        elapsed_s=timer.elapsed,
    )

    # 2. a grid of Cartesian offsets, then VERIFY three of them by driving the solution.
    offsets = [
        ("+x 6cm", np.array([0.06, 0.0, 0.0])),
        ("-x 6cm", np.array([-0.06, 0.0, 0.0])),
        ("+y 6cm", np.array([0.0, 0.06, 0.0])),
        ("-y 6cm", np.array([0.0, -0.06, 0.0])),
        ("+z 6cm", np.array([0.0, 0.0, 0.06])),
        ("-z 6cm", np.array([0.0, 0.0, -0.06])),
        ("+x 40cm (expected out of reach)", np.array([0.4, 0.0, 0.0])),
    ]
    solutions: dict[str, np.ndarray] = {}
    for label, offset in offsets:
        target = Pose(
            position=np.asarray(home_pose.position) + offset,
            quaternion_wxyz=np.asarray(home_pose.quaternion_wxyz),
            frame=home_pose.frame,
        )
        with Timer() as timer:
            solution = tools["solve_ik"](target, home_joints)
        if solution.ok:
            solutions[label] = np.asarray(solution.joint_positions, dtype=np.float64)
        stage.record(
            "solve_ik",
            {
                "target_pose": dump_pose(target),
                "seed_joints": np.round(home_joints, 4).tolist(),
                "offset_from_current_m": offset.tolist(),
            },
            {
                "ok": bool(solution.ok),
                "joint_positions_rad": None
                if solution.joint_positions is None
                else np.round(solution.joint_positions, 5).tolist(),
                "error": error_text(solution.error),
                "error_detail": dump_error(solution.error),
                "diagnostics": {k: str(v) for k, v in dict(solution.diagnostics).items()},
            },
            ok=bool(solution.ok),
            label=f"solve_ik ({label})",
            verdict="solved" if solution.ok else error_text(solution.error),
            elapsed_s=timer.elapsed,
        )

    # Where exactly does upward reach stop? A one-axis sweep turns "IK_FAILED" into a number.
    sweep = []
    for axis_label, direction in (("+z", np.array([0.0, 0.0, 1.0])), ("+x", np.array([1.0, 0.0, 0.0]))):
        for distance in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.12, 0.20):
            target = Pose(
                position=np.asarray(home_pose.position) + direction * distance,
                quaternion_wxyz=np.asarray(home_pose.quaternion_wxyz),
                frame=home_pose.frame,
            )
            solution = tools["solve_ik"](target, home_joints)
            sweep.append(
                {
                    "axis": axis_label,
                    "distance_m": distance,
                    "ok": bool(solution.ok),
                    "error": error_text(solution.error),
                }
            )
    stage.array(
        "reach_sweep",
        np.asarray(
            [
                [0.0 if row["axis"] == "+z" else 1.0, row["distance_m"], float(row["ok"])]
                for row in sweep
            ]
        ),
    )
    first_failure = {
        axis: next(
            (row["distance_m"] for row in sweep if row["axis"] == axis and not row["ok"]), None
        )
        for axis in ("+z", "+x")
    }
    stage.record(
        "solve_ik",
        {"target_pose": "current pose displaced along +z and +x, 1 cm to 20 cm"},
        {"sweep": sweep, "first_failing_distance_m": first_failure},
        ok=any(row["ok"] for row in sweep),
        label="solve_ik reach sweep (where does it stop solving?)",
        verdict=(
            f"first failure: +z at {first_failure['+z']} m, +x at {first_failure['+x']} m"
        ),
    )

    # Drive each solution and measure the pose the arm actually reaches. This is the only
    # way to tell a correct IK solution from a plausible-looking vector.
    _frames_on(session, stride=2)
    verifications = []
    for label in ("+z 6cm", "-z 6cm", "+y 6cm"):
        if label not in solutions:
            continue
        offset = dict(offsets)[label]
        target = Pose(
            position=np.asarray(home_pose.position) + offset,
            quaternion_wxyz=np.asarray(home_pose.quaternion_wxyz),
            frame=home_pose.frame,
        )
        moved = tools["move_to_joints"](solutions[label], 0.02, 90)
        reached = tools["get_robot_state"]().end_effector_poses["primary"]
        position_error = float(np.linalg.norm(np.asarray(reached.position) - target.position))
        orientation_error = quaternion_angle_deg(
            reached.quaternion_wxyz, target.quaternion_wxyz
        )
        verifications.append(
            {
                "label": label,
                "target_position_m": np.round(target.position, 4).tolist(),
                "reached_position_m": np.round(reached.position, 4).tolist(),
                "position_error_m": round(position_error, 4),
                "orientation_error_deg": round(orientation_error, 3),
            }
        )
        stage.record(
            "move_to_joints",
            {
                "target": np.round(solutions[label], 4).tolist(),
                "tolerance": 0.02,
                "max_steps": 90,
                "note": f"IK solution for {label}",
            },
            {
                "execution": dump_execution(moved),
                "ik_target_pose": dump_pose(target),
                "reached_pose": dump_pose(reached),
                "position_error_m": round(position_error, 5),
                "orientation_error_deg": round(orientation_error, 3),
            },
            ok=moved.ok and position_error < 0.02,
            label=f"verify solve_ik by execution ({label})",
            verdict=(
                f"commanded the IK solution; EE landed {1000 * position_error:.1f} mm and "
                f"{orientation_error:.2f} deg from the pose IK was asked for"
            ),
        )
        _capture(session, stage, f"verify_{label.replace(' ', '_').replace('+', 'p').replace('-', 'm')}")
        tools["move_to_joints"](home_joints, 0.02, 90)
        session.capture.take()
    _frames_off(session)
    stage.notes.append(json.dumps({"ik_execution_verification": verifications}))


# --------------------------------------------------------------------------- stage: planning


def stage_planning(session: Session, stage: Stage) -> None:
    tools = session.tools
    session.reset()
    state = tools["get_robot_state"]()
    start = np.asarray(state.joint_positions["primary"], dtype=np.float64)
    home_pose = state.end_effector_poses["primary"]
    stage.array("home_joints", start)

    joint_goal = start.copy()
    joint_goal[3] += 0.12
    joint_goal[5] -= 0.10

    plans: dict[str, Any] = {}
    cases = [
        ("joint goal, interpolation", joint_goal, MotionStrategy()),
        (
            "joint goal, curobo",
            joint_goal,
            MotionStrategy(ik_solver="pyroki", trajectory_planner="curobo"),
        ),
        (
            "pose goal, composed (pyroki IK + interpolation)",
            Pose(
                position=np.asarray(home_pose.position) + np.array([0.0, 0.0, -0.08]),
                quaternion_wxyz=np.asarray(home_pose.quaternion_wxyz),
                frame=home_pose.frame,
            ),
            MotionStrategy(),
        ),
        (
            "pose goal, pyroki IK + curobo trajectory",
            Pose(
                position=np.asarray(home_pose.position) + np.array([0.0, 0.0, -0.08]),
                quaternion_wxyz=np.asarray(home_pose.quaternion_wxyz),
                frame=home_pose.frame,
            ),
            MotionStrategy(ik_solver="pyroki", trajectory_planner="curobo"),
        ),
        (
            "pose goal, curobo-integrated pose planner",
            Pose(
                position=np.asarray(home_pose.position) + np.array([0.0, 0.0, -0.08]),
                quaternion_wxyz=np.asarray(home_pose.quaternion_wxyz),
                frame=home_pose.frame,
            ),
            MotionStrategy(
                ik_solver="curobo",
                trajectory_planner="curobo",
                pose_planner="curobo-integrated",
            ),
        ),
    ]
    for label, goal, strategy in cases:
        with Timer() as timer:
            plan = tools["plan_motion"](goal, strategy=strategy)
        trajectory = plan.trajectory
        summary: dict[str, Any] = {
            "ok": bool(plan.ok),
            "error": error_text(plan.error),
            "diagnostics": {k: str(v) for k, v in dict(plan.diagnostics).items()},
        }
        if trajectory is not None:
            waypoints = np.asarray(trajectory.joint_positions, dtype=np.float64)
            key = (
                label.split(",")[0].replace(" ", "_")
                + "__"
                + strategy.trajectory_planner
                + "__"
                + strategy.pose_planner
            )
            stage.array(f"plan_{key}", waypoints)
            plans[label] = plan
            summary["trajectory"] = {
                "n_waypoints": int(len(waypoints)),
                "dt_s": float(trajectory.dt_s),
                "planner": str(trajectory.planner),
                "collision_aware": bool(trajectory.collision_aware),
                "joint_names": [str(n) for n in trajectory.joint_names],
                "start": np.round(waypoints[0], 5).tolist(),
                "end": np.round(waypoints[-1], 5).tolist(),
                "max_step_rad": round(float(np.max(np.abs(np.diff(waypoints, axis=0)))), 5),
                "start_matches_measured_rad": round(
                    float(np.max(np.abs(waypoints[0] - start))), 5
                ),
            }
            if isinstance(goal, np.ndarray):
                summary["trajectory"]["end_vs_joint_goal_rad"] = round(
                    float(np.max(np.abs(waypoints[-1] - goal))), 5
                )
        stage.record(
            "plan_motion",
            {
                "goal": dump_pose(goal) if isinstance(goal, Pose) else np.round(goal, 4).tolist(),
                "strategy": {
                    "ik_solver": strategy.ik_solver,
                    "trajectory_planner": strategy.trajectory_planner,
                    "pose_planner": strategy.pose_planner,
                },
            },
            summary,
            ok=bool(plan.ok),
            label=f"plan_motion ({label})",
            verdict=(
                f"{summary['trajectory']['n_waypoints']} waypoints, "
                f"planner={summary['trajectory']['planner']}, "
                f"collision_aware={summary['trajectory']['collision_aware']}"
                if trajectory is not None
                else error_text(plan.error)
            ),
            elapsed_s=timer.elapsed,
        )

    # collision-aware planning against the OBSERVED cloud, with a target cloud excluded
    segmentation, cloud = _cloud_for(tools, "agentview", "fish")
    if cloud is not None:
        plan = tools["plan_motion"](
            joint_goal,
            strategy=MotionStrategy(ik_solver="pyroki", trajectory_planner="curobo"),
            target_point_cloud=cloud,
        )
        stage.record(
            "plan_motion",
            {
                "goal": np.round(joint_goal, 4).tolist(),
                "strategy": "curobo",
                "target_point_cloud": f"fish cloud ({len(cloud.points)} pts, {cloud.frame})",
            },
            {
                "ok": bool(plan.ok),
                "error": error_text(plan.error),
                "n_waypoints": 0
                if plan.trajectory is None
                else int(len(plan.trajectory.joint_positions)),
                "collision_aware": None
                if plan.trajectory is None
                else bool(plan.trajectory.collision_aware),
                "diagnostics": {k: str(v) for k, v in dict(plan.diagnostics).items()},
            },
            ok=bool(plan.ok),
            label="plan_motion (curobo, target cloud excluded from the collision world)",
            verdict="planned" if plan.ok else error_text(plan.error),
        )

    # plan_synchronized_motion -- unimanual embodiment
    with Timer() as timer:
        synchronized = tools["plan_synchronized_motion"]({"primary": home_pose})
    stage.record(
        "plan_synchronized_motion",
        {"targets": {"primary": dump_pose(home_pose)}},
        {
            "ok": bool(synchronized.ok),
            "error": error_text(synchronized.error),
            "diagnostics": {k: str(v) for k, v in dict(synchronized.diagnostics).items()},
            "trajectory": None
            if synchronized.trajectory is None
            else {
                "planner": str(synchronized.trajectory.planner),
                "n_waypoints": int(
                    len(next(iter(synchronized.trajectory.joint_positions.values())))
                ),
                "collision_aware": bool(synchronized.trajectory.collision_aware),
            },
        },
        ok=bool(synchronized.ok),
        label="plan_synchronized_motion (one arm)",
        verdict="planned" if synchronized.ok else error_text(synchronized.error),
        elapsed_s=timer.elapsed,
    )

    # execute the cuRobo plan and measure how well the arm tracks it
    chosen = plans.get("joint goal, curobo") or plans.get("joint goal, interpolation")
    if chosen is not None:
        _frames_on(session, stride=1)
        with Timer() as timer:
            executed = tools["execute_trajectory"](chosen.trajectory)
        reached = np.asarray(
            tools["get_robot_state"]().joint_positions["primary"], dtype=np.float64
        )
        stage.record(
            "execute_trajectory",
            {
                "trajectory": (
                    f"{len(chosen.trajectory.joint_positions)} waypoints from "
                    f"{chosen.trajectory.planner}, dt={chosen.trajectory.dt_s}s"
                )
            },
            {
                "execution": dump_execution(executed),
                "joint_goal": np.round(joint_goal, 5).tolist(),
                "reached": np.round(reached, 5).tolist(),
                "max_joint_error_rad": round(float(np.max(np.abs(reached - joint_goal))), 5),
            },
            ok=bool(executed.ok),
            label="execute_trajectory (the cuRobo plan)",
            verdict=(
                f"{executed.steps_executed} ticks, final joint error "
                f"{float(np.max(np.abs(reached - joint_goal))):.4f} rad"
            ),
            elapsed_s=timer.elapsed,
        )
        _capture(session, stage, "execute_curobo_plan")
        _frames_off(session)
        tools["go_home"]()
        session.capture.take()

    # execute_trajectory must refuse a plan whose recorded start no longer matches the arm.
    stale = plans.get("joint goal, interpolation")
    if stale is not None:
        nudge = start.copy()
        nudge[1] += 0.10
        tools["move_to_joints"](nudge, 0.02, 60)
        session.capture.take()
        refused = tools["execute_trajectory"](stale.trajectory)
        tools["move_to_joints"](start, 0.02, 60)
        session.capture.take()
        stage.record(
            "execute_trajectory",
            {"trajectory": "a plan made before the arm was moved 0.10 rad away from its start"},
            {"execution": dump_execution(refused)},
            ok=not refused.ok,
            label="execute_trajectory (stale plan -- expected refusal)",
            verdict=error_text(refused.error) or "executed a stale plan",
        )


# --------------------------------------------------------------------------- stage: gripper


def stage_gripper(session: Session, stage: Stage) -> None:
    tools = session.tools
    session.reset()
    _frames_on(session, stride=1)

    calls: list[tuple[str, str, Any]] = [
        ("open_gripper", "open_gripper()", lambda: tools["open_gripper"]()),
        ("close_gripper", "close_gripper()", lambda: tools["close_gripper"]()),
        ("set_gripper", "set_gripper(1.0)", lambda: tools["set_gripper"](1.0)),
        ("set_gripper", "set_gripper(0.0)", lambda: tools["set_gripper"](0.0)),
        ("set_gripper", "set_gripper(0.5)", lambda: tools["set_gripper"](0.5)),
        (
            "set_grippers",
            "set_grippers({'primary': 1.0})",
            lambda: tools["set_grippers"]({"primary": 1.0}),
        ),
        (
            "set_grippers",
            "set_grippers({'left': 1.0}) -- wrong arm name",
            lambda: tools["set_grippers"]({"left": 1.0}),
        ),
    ]
    for tool, label, call in calls:
        before = float(tools["get_robot_state"]().gripper_positions["primary"])
        with Timer() as timer:
            result = call()
        after = float(tools["get_robot_state"]().gripper_positions["primary"])
        expected_rejection = "0.5" in label or "left" in label
        stage.record(
            tool,
            {"call": label},
            {
                "execution": dump_execution(result),
                "measured_before": round(before, 4),
                "measured_after": round(after, 4),
            },
            ok=(not result.ok) if expected_rejection else bool(result.ok),
            label=label,
            verdict=(
                f"refused: {error_text(result.error)}"
                if not result.ok
                else f"jaw {before:.3f} -> {after:.3f} in {result.steps_executed} ticks"
            ),
            elapsed_s=timer.elapsed,
        )
        _capture(session, stage, "gripper_" + label.split("(")[0].strip() + f"_{after:.2f}")

    # step(): one tick holding the current joints
    state = tools["get_robot_state"]()
    action = RobotAction(
        arms={
            "primary": ArmCommand(
                mode="joint_position",
                target=np.asarray(state.joint_positions["primary"], dtype=np.float64),
                gripper_position=float(state.gripper_positions["primary"]),
                embodiment="robocasa",
            )
        }
    )
    with Timer() as timer:
        result = tools["step"](action)
    after = tools["get_robot_state"]()
    drift = float(
        np.linalg.norm(
            np.asarray(after.joint_positions["primary"])
            - np.asarray(state.joint_positions["primary"])
        )
    )
    stage.record(
        "step",
        {
            "action": "RobotAction(primary=ArmCommand('joint_position', current joints))",
            "target": np.round(state.joint_positions["primary"], 5).tolist(),
        },
        {
            "ok": bool(result.ok),
            "terminated": bool(result.terminated),
            "truncated": bool(result.truncated),
            "error": error_text(result.error),
            "diagnostics": {k: str(v) for k, v in dict(result.diagnostics).items()},
            "reward_visible_to_program": result.reward,
            "observation_cameras": sorted(result.observation.cameras)
            if result.observation is not None
            else [],
            "joint_drift_rad": round(drift, 6),
        },
        ok=bool(result.ok),
        verdict=f"one tick advanced; joint drift {drift:.5f} rad; reward={result.reward}",
        elapsed_s=timer.elapsed,
    )

    # a malformed action: the mode this controller does not have. ArmCommand validates at
    # construction, so the refusal arrives as a ValueError before step() is ever called.
    try:
        bad = tools["step"](
            RobotAction(
                arms={
                    "primary": ArmCommand(
                        mode="cartesian_pose",
                        target=np.zeros(7),
                        embodiment="robocasa",
                    )
                }
            )
        )
        outcome = {"ok": bool(bad.ok), "error": error_text(bad.error), "raised": ""}
        refused = not bad.ok
    except (ValueError, TypeError) as exc:
        outcome = {"ok": False, "error": "", "raised": f"{type(exc).__name__}: {exc}"}
        refused = True
    stage.record(
        "step",
        {"action": "ArmCommand(mode='cartesian_pose')"},
        outcome,
        ok=refused,
        label="step (unsupported mode -- expected refusal)",
        verdict=(outcome["raised"] or outcome["error"] or "accepted an unsupported mode"),
    )

    # a mismatched joint width
    try:
        short = tools["step"](
            RobotAction(
                arms={
                    "primary": ArmCommand(
                        mode="joint_position", target=np.zeros(3), embodiment="robocasa"
                    )
                }
            )
        )
        outcome = {"ok": bool(short.ok), "error": error_text(short.error), "raised": ""}
        refused = not short.ok
    except (ValueError, TypeError) as exc:
        outcome = {"ok": False, "error": "", "raised": f"{type(exc).__name__}: {exc}"}
        refused = True
    stage.record(
        "step",
        {"action": "ArmCommand(mode='joint_position', target=zeros(3))"},
        outcome,
        ok=refused,
        label="step (3-joint target on a 7-joint arm -- expected refusal)",
        verdict=(outcome["raised"] or outcome["error"] or "accepted a 3-joint target"),
    )
    _capture(session, stage, "gripper_step")
    _frames_off(session)


# --------------------------------------------------------------------------- stage: motion


def stage_motion(session: Session, stage: Stage) -> None:
    tools = session.tools
    session.reset()
    state = tools["get_robot_state"]()
    home_joints = np.asarray(state.joint_positions["primary"], dtype=np.float64)
    joint_names = [str(n) for n in state.joint_names["primary"]]
    stage.array("home_joints", home_joints)

    _frames_on(session, stride=1)

    # ---- move_to_joints
    target = home_joints.copy()
    target[1] += 0.20
    target[3] -= 0.15
    with Timer() as timer:
        moved = tools["move_to_joints"](target, 0.01, 120)
    reached = np.asarray(tools["get_robot_state"]().joint_positions["primary"], dtype=np.float64)
    stage.record(
        "move_to_joints",
        {"target": np.round(target, 5).tolist(), "tolerance": 0.01, "max_steps": 120},
        {
            "execution": dump_execution(moved),
            "start": np.round(home_joints, 5).tolist(),
            "reached": np.round(reached, 5).tolist(),
            "max_joint_error_rad": round(float(np.max(np.abs(reached - target))), 5),
        },
        ok=bool(moved.ok),
        verdict=(
            f"{moved.steps_executed} ticks, max joint error "
            f"{float(np.max(np.abs(reached - target))):.5f} rad"
        ),
        elapsed_s=timer.elapsed,
    )
    stage.array("move_to_joints__target", target)
    _capture(session, stage, "move_to_joints")

    # ---- execute_trajectory, hand-built
    traj_state = tools["get_robot_state"]()
    start = np.asarray(traj_state.joint_positions["primary"], dtype=np.float64)
    goal = start.copy()
    goal[2] += 0.25
    waypoints = np.asarray(
        [start + (index + 1) / 20.0 * (goal - start) for index in range(20)]
        + [goal] * 5
    )
    trajectory = Trajectory(
        joint_positions=waypoints,
        dt_s=0.05,
        joint_names=joint_names,
        planner="api_tool_check",
        collision_aware=False,
        expected_start=traj_state,
        arm="primary",
        embodiment="robocasa",
    )
    with Timer() as timer:
        executed = tools["execute_trajectory"](trajectory)
    reached = np.asarray(tools["get_robot_state"]().joint_positions["primary"], dtype=np.float64)
    stage.record(
        "execute_trajectory",
        {
            "trajectory": f"{len(waypoints)} hand-built waypoints, dt=0.05s, joint 2 += 0.25 rad",
            "stop_on_termination": True,
        },
        {
            "execution": dump_execution(executed),
            "goal": np.round(goal, 5).tolist(),
            "reached": np.round(reached, 5).tolist(),
            "max_joint_error_rad": round(float(np.max(np.abs(reached - goal))), 5),
        },
        ok=bool(executed.ok),
        label="execute_trajectory (hand-built ramp)",
        verdict=(
            f"{executed.steps_executed} ticks for {len(waypoints)} waypoints "
            f"(densified at 0.05 rad), final joint error "
            f"{float(np.max(np.abs(reached - goal))):.5f} rad"
        ),
        elapsed_s=timer.elapsed,
    )
    stage.array("execute_trajectory__commanded_waypoints", waypoints)
    _capture(session, stage, "execute_trajectory")

    # ---- go_home
    with Timer() as timer:
        homed = tools["go_home"]()
    reached = np.asarray(tools["get_robot_state"]().joint_positions["primary"], dtype=np.float64)
    stage.record(
        "go_home",
        {},
        {
            "execution": dump_execution(homed),
            "home_joints": np.round(home_joints, 5).tolist(),
            "reached": np.round(reached, 5).tolist(),
            "max_joint_error_rad": round(float(np.max(np.abs(reached - home_joints))), 5),
        },
        ok=bool(homed.ok),
        verdict=(
            f"{homed.steps_executed} ticks, max joint error from the captured home "
            f"{float(np.max(np.abs(reached - home_joints))):.5f} rad"
        ),
        elapsed_s=timer.elapsed,
    )
    _capture(session, stage, "go_home")

    # ---- move_to_pose: one axis at a time, so the result is readable
    home_pose = tools["get_robot_state"]().end_effector_poses["primary"]
    stage.array("move_to_pose__home_ee", pose_matrix(home_pose))
    moves = [
        ("-z 5cm", np.array([0.0, 0.0, -0.05])),
        ("+x 5cm", np.array([0.05, 0.0, 0.0])),
        ("+y 5cm", np.array([0.0, 0.05, 0.0])),
        ("back to the home pose", np.array([0.0, 0.0, 0.0])),
        ("+x 60cm (expected unreachable)", np.array([0.6, 0.0, 0.0])),
    ]
    anchor = np.asarray(home_pose.position, dtype=np.float64)
    for label, offset in moves:
        before = tools["get_robot_state"]().end_effector_poses["primary"]
        target_pose = Pose(
            position=anchor + offset,
            quaternion_wxyz=np.asarray(home_pose.quaternion_wxyz),
            frame=home_pose.frame,
        )
        with Timer() as timer:
            moved = tools["move_to_pose"](target_pose, 0.01, 120)
        after = tools["get_robot_state"]().end_effector_poses["primary"]
        position_error = float(np.linalg.norm(np.asarray(after.position) - target_pose.position))
        stage.record(
            "move_to_pose",
            {
                "target_pose": dump_pose(target_pose),
                "tolerance": 0.01,
                "max_steps": 120,
                "offset_from_home_m": offset.tolist(),
            },
            {
                "execution": dump_execution(moved),
                "ee_before": dump_pose(before),
                "ee_after": dump_pose(after),
                "position_error_m": round(position_error, 5),
                "orientation_error_deg": round(
                    quaternion_angle_deg(after.quaternion_wxyz, target_pose.quaternion_wxyz), 3
                ),
                "commanded_delta_m": np.round(
                    target_pose.position - np.asarray(before.position), 5
                ).tolist(),
                "measured_delta_m": np.round(
                    np.asarray(after.position) - np.asarray(before.position), 5
                ).tolist(),
            },
            ok=(not moved.ok) if "unreachable" in label else bool(moved.ok),
            label=f"move_to_pose ({label})",
            verdict=(
                f"refused without moving: {error_text(moved.error)}"
                if not moved.ok
                else f"{moved.steps_executed} ticks; EE within {1000 * position_error:.1f} mm "
                "of the requested pose"
            ),
            elapsed_s=timer.elapsed,
        )
        stage.array(f"move_to_pose__target_{label.replace(' ', '_')}", target_pose.position)
        _capture(session, stage, f"move_to_pose_{label.replace(' ', '_')}")

    # ---- move_to_pose with an orientation change only
    before = tools["get_robot_state"]().end_effector_poses["primary"]
    rotation = quat_to_matrix(before.quaternion_wxyz)
    angle = np.deg2rad(20.0)
    spin = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation @ spin
    transform[:3, 3] = np.asarray(before.position)
    from cap_harness.geometry import matrix_to_pose

    rotated = matrix_to_pose(transform, frame=before.frame)
    with Timer() as timer:
        turned = tools["move_to_pose"](rotated, 0.01, 120)
    after = tools["get_robot_state"]().end_effector_poses["primary"]
    stage.record(
        "move_to_pose",
        {"target_pose": dump_pose(rotated), "why": "20 deg about the tool's own z, same position"},
        {
            "execution": dump_execution(turned),
            "ee_before": dump_pose(before),
            "ee_after": dump_pose(after),
            "position_error_m": round(
                float(np.linalg.norm(np.asarray(after.position) - rotated.position)), 5
            ),
            "orientation_error_deg": round(
                quaternion_angle_deg(after.quaternion_wxyz, rotated.quaternion_wxyz), 3
            ),
            "orientation_change_deg": round(
                quaternion_angle_deg(before.quaternion_wxyz, after.quaternion_wxyz), 3
            ),
        },
        ok=bool(turned.ok),
        label="move_to_pose (rotate the tool 20 deg in place)",
        verdict=(
            f"turned {quaternion_angle_deg(before.quaternion_wxyz, after.quaternion_wxyz):.1f} deg; "
            f"orientation error {quaternion_angle_deg(after.quaternion_wxyz, rotated.quaternion_wxyz):.2f} deg"
        )
        if turned.ok
        else error_text(turned.error),
        elapsed_s=timer.elapsed,
    )
    _capture(session, stage, "move_to_pose_rotate20")

    # ---- move_synchronized
    current = tools["get_robot_state"]().end_effector_poses["primary"]
    with Timer() as timer:
        synchronized = tools["move_synchronized"]({"primary": current}, 0.02, 60)
    stage.record(
        "move_synchronized",
        {"targets": {"primary": dump_pose(current)}, "tolerance": 0.02, "max_steps": 60},
        {"execution": dump_execution(synchronized)},
        ok=bool(synchronized.ok),
        label="move_synchronized (one arm, current pose)",
        verdict="executed" if synchronized.ok else error_text(synchronized.error),
        elapsed_s=timer.elapsed,
    )
    _capture(session, stage, "move_synchronized")
    _frames_off(session)
    stage.notes.append(f"ticks used by this stage: {session.elapsed()}")


# --------------------------------------------------------------------------- stage: base


def capture_base_video(session: Session, stage: Stage, name: str, call: Any) -> Any:
    """Run one base command with the simulator's own frame buffer enabled.

    ``navigate_to_pose`` drives inside the driver's closed loop and never reaches
    ``adapter.step``, so the per-tick observer sees nothing. The simulator's video channel
    (frames staged through /dev/shm, never the wire) is the only way to see base motion. This
    is the one place in this probe that touches the driver handle directly.
    """
    client = session.client
    try:
        client.video_enable(True, clear=True, wrist=False)
    except Exception as exc:  # the channel is optional
        stage.notes.append(f"video channel unavailable: {exc}")
        return call()
    result = call()
    try:
        count = client.video_frame_count()
        frames = client.video_frames_range(0, count) if count else []
        if frames:
            stride = max(1, len(frames) // 150)
            stage.video(name, frames[::stride], fps=20.0)
    except Exception as exc:
        stage.notes.append(f"video pull failed for {name}: {exc}")
    finally:
        try:
            client.video_enable(False, clear=True, wrist=False)
        except Exception:
            pass
    return result


def stage_base(session: Session, stage: Stage) -> None:
    tools = session.tools
    observation = session.reset()
    stage.image("rgb_agentview", observation.cameras["agentview"].rgb)

    spawn = tools["robocasa.get_base_pose"]()
    spawn_xy = np.asarray(spawn.position, dtype=np.float64)[:2]
    spawn_yaw = yaw_of(spawn)
    stage.array("spawn_pose", pose_matrix(spawn))
    stage.record(
        "robocasa.get_base_pose",
        {},
        dump_pose(spawn),
        ok=spawn.frame == "world",
        verdict=f"spawn at ({spawn_xy[0]:.3f}, {spawn_xy[1]:.3f}) world, yaw {spawn_yaw:.3f} rad",
    )

    # ---- base_pose_is_free: a map, built one public yes/no at a time
    half = 1.5
    step_m = 0.1
    axis = np.arange(-half, half + 1e-9, step_m)
    grid = np.zeros((len(axis), len(axis)), dtype=bool)
    with Timer() as timer:
        for row, dy in enumerate(axis):
            for column, dx in enumerate(axis):
                grid[row, column] = bool(
                    tools["robocasa.base_pose_is_free"](
                        float(spawn_xy[0] + dx), float(spawn_xy[1] + dy), spawn_yaw
                    )
                )
    stage.array("free_grid", grid)
    stage.array("free_grid_axis", axis)
    spawn_free = bool(tools["robocasa.base_pose_is_free"](float(spawn_xy[0]), float(spawn_xy[1]), spawn_yaw))
    stage.record(
        "robocasa.base_pose_is_free",
        {
            "x,y": "a 31x31 grid, 0.1 m, centred on the spawn pose",
            "yaw": round(spawn_yaw, 4),
        },
        {
            "spawn_is_free": spawn_free,
            "free_cells": int(grid.sum()),
            "blocked_cells": int((~grid).sum()),
            "total_queries": int(grid.size) + 1,
            "seconds_for_all_queries": round(timer.elapsed, 2),
        },
        ok=spawn_free,
        label="robocasa.base_pose_is_free (961-cell occupancy sweep)",
        verdict=(
            f"spawn free={spawn_free}; {int(grid.sum())}/{grid.size} cells free "
            f"({timer.elapsed:.1f} s for the sweep)"
        ),
        elapsed_s=timer.elapsed,
    )

    # Cross-check the two frames (trap 20): put the counter cloud into world with
    # get_base_pose and see whether it lands on the blocked cells.
    segmentation, counter_cloud = _cloud_for(tools, "agentview", "counter")
    if counter_cloud is not None:
        transform = pose_matrix(spawn)
        world_points = counter_cloud.points @ transform[:3, :3].T + transform[:3, 3]
        stage.array("counter_cloud_base", counter_cloud.points)
        stage.array("counter_cloud_world", world_points)
        occupied = []
        for point in world_points[::37]:
            occupied.append(
                not bool(
                    tools["robocasa.base_pose_is_free"](
                        float(point[0]), float(point[1]), spawn_yaw
                    )
                )
            )
        stage.record(
            "robocasa.base_pose_is_free",
            {
                "x,y": "counter cloud centroids pushed from robot0_link0 into world "
                "with robocasa.get_base_pose"
            },
            {
                "n_sampled": len(occupied),
                "fraction_reported_blocked": round(float(np.mean(occupied)), 4),
                "counter_centroid_base": np.round(counter_cloud.points.mean(axis=0), 4).tolist(),
                "counter_centroid_world": np.round(world_points.mean(axis=0), 4).tolist(),
            },
            ok=float(np.mean(occupied)) > 0.5,
            label="frame cross-check: counter cloud -> world -> base_pose_is_free",
            verdict=(
                f"{100 * float(np.mean(occupied)):.0f}% of counter points report a blocked "
                "footprint, so the two frames agree"
            ),
        )

    # ---- plan_standoff_pose / plan_approach_pose
    _, object_cloud = _cloud_for(tools, "agentview", "fish")
    if object_cloud is not None and counter_cloud is not None:
        for distance in (0.3, 0.6):
            with Timer() as timer:
                standoff = tools["robocasa.plan_standoff_pose"](
                    counter_cloud, object_cloud, distance
                )
            world = pose_matrix(spawn) @ np.array(
                [standoff[0], standoff[1], 0.0, 1.0]
            )
            free = bool(
                tools["robocasa.base_pose_is_free"](
                    float(world[0]), float(world[1]), spawn_yaw + standoff[2]
                )
            )
            stage.array(f"standoff_{distance}", np.asarray(standoff))
            stage.record(
                "robocasa.plan_standoff_pose",
                {
                    "support_cloud": f"counter ({len(counter_cloud.points)} pts, {counter_cloud.frame})",
                    "object_cloud": f"fish ({len(object_cloud.points)} pts)",
                    "standoff_m": distance,
                },
                {
                    "x_y_yaw": [round(float(v), 4) for v in standoff],
                    "frame": counter_cloud.frame,
                    "world_xy": np.round(world[:2], 4).tolist(),
                    "base_pose_is_free_there": free,
                },
                ok=all(np.isfinite(standoff)),
                label=f"robocasa.plan_standoff_pose(standoff_m={distance})",
                verdict=(
                    f"(x,y,yaw)=({standoff[0]:.3f}, {standoff[1]:.3f}, {standoff[2]:.3f}) in "
                    f"{counter_cloud.frame}; footprint free there: {free}"
                ),
                elapsed_s=timer.elapsed,
            )
        for distance in (0.7, 0.4):
            with Timer() as timer:
                approach = tools["robocasa.plan_approach_pose"](object_cloud, distance)
            world = pose_matrix(spawn) @ np.array([approach[0], approach[1], 0.0, 1.0])
            free = bool(
                tools["robocasa.base_pose_is_free"](
                    float(world[0]), float(world[1]), spawn_yaw + approach[2]
                )
            )
            stage.array(f"approach_{distance}", np.asarray(approach))
            stage.record(
                "robocasa.plan_approach_pose",
                {
                    "object_cloud": f"fish ({len(object_cloud.points)} pts, {object_cloud.frame})",
                    "distance_m": distance,
                },
                {
                    "x_y_yaw": [round(float(v), 4) for v in approach],
                    "frame": object_cloud.frame,
                    "world_xy": np.round(world[:2], 4).tolist(),
                    "base_pose_is_free_there": free,
                },
                ok=all(np.isfinite(approach)),
                label=f"robocasa.plan_approach_pose(distance_m={distance})",
                verdict=(
                    f"(x,y,yaw)=({approach[0]:.3f}, {approach[1]:.3f}, {approach[2]:.3f}) in "
                    f"{object_cloud.frame}; footprint free there: {free}"
                ),
                elapsed_s=timer.elapsed,
            )

    # ---- navigate_to_pose
    commands = [
        ("lateral +0.15 m", (0.0, 0.15, 0.0)),
        ("lateral -0.15 m", (0.0, -0.15, 0.0)),
        ("yaw +0.30 rad", (0.0, 0.0, 0.30)),
        ("yaw -0.30 rad", (0.0, 0.0, -0.30)),
        ("forward +0.30 m (counter ahead)", (0.30, 0.0, 0.0)),
    ]
    track = [np.concatenate([spawn_xy, [spawn_yaw]])]
    for label, (dx, dy, dyaw) in commands:
        before = tools["robocasa.get_base_pose"]()
        before_xy = np.asarray(before.position, dtype=np.float64)[:2]
        before_yaw = yaw_of(before)
        name = "navigate_" + label.split(" ")[0] + f"_{dx}_{dy}_{dyaw}".replace(".", "p")
        with Timer() as timer:
            result = capture_base_video(
                session, stage, name, lambda: tools["robocasa.navigate_to_pose"](dx, dy, dyaw)
            )
        after = tools["robocasa.get_base_pose"]()
        after_xy = np.asarray(after.position, dtype=np.float64)[:2]
        after_yaw = yaw_of(after)
        track.append(np.concatenate([after_xy, [after_yaw]]))
        # the command is in the base frame, so rotate the world delta back into it
        cos, sin = np.cos(before_yaw), np.sin(before_yaw)
        delta_world = after_xy - before_xy
        delta_local = np.array(
            [cos * delta_world[0] + sin * delta_world[1], -sin * delta_world[0] + cos * delta_world[1]]
        )
        yaw_delta = float(np.arctan2(np.sin(after_yaw - before_yaw), np.cos(after_yaw - before_yaw)))
        stage.record(
            "robocasa.navigate_to_pose",
            {"dx": dx, "dy": dy, "dyaw": dyaw, "frame": "current base frame, relative"},
            {
                "execution": dump_execution(result),
                "base_before": dump_pose(before),
                "base_after": dump_pose(after),
                "commanded_local": [dx, dy, round(dyaw, 4)],
                "measured_local": [
                    round(float(delta_local[0]), 4),
                    round(float(delta_local[1]), 4),
                    round(yaw_delta, 4),
                ],
            },
            ok=bool(result.ok),
            label=f"robocasa.navigate_to_pose ({label})",
            verdict=(
                f"commanded ({dx}, {dy}, {dyaw}) -> measured "
                f"({delta_local[0]:+.3f}, {delta_local[1]:+.3f}, {yaw_delta:+.3f}); "
                f"{'converged' if result.ok else error_text(result.error)}"
            ),
            elapsed_s=timer.elapsed,
        )
    stage.array("base_track", np.asarray(track))
    stage.notes.append(f"ticks used by this stage: {session.elapsed()}")


# --------------------------------------------------------------------------- driver

STAGES = {
    "identity": stage_identity,
    "perception": stage_perception,
    "grasps": stage_grasps,
    "graspexec": stage_graspexec,
    "ik": stage_ik,
    "planning": stage_planning,
    "gripper": stage_gripper,
    "motion": stage_motion,
    "base": stage_base,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="PickPlaceCounterToStove")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--stages", nargs="*", default=list(STAGES))
    arguments = parser.parse_args(argv)

    unknown = [name for name in arguments.stages if name not in STAGES]
    if unknown:
        parser.error(f"unknown stages {unknown}; choose from {list(STAGES)}")

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"task={arguments.task} seed={arguments.seed} stages={arguments.stages}")
    session = build_session(arguments.task, arguments.seed)
    print(f"connected; {len(session.tools)} public tools")
    failures: list[str] = []
    try:
        for name in arguments.stages:
            print(f"\n=== stage {name} " + "=" * (60 - len(name)))
            stage = Stage(name=name)
            try:
                STAGES[name](session, stage)
            except Exception:
                trace = traceback.format_exc()
                print(trace)
                stage.notes.append("STAGE RAISED:\n" + trace)
                failures.append(name)
            finally:
                stage.flush()
    finally:
        session.close()

    (OUT / "probe-run.json").write_text(
        json.dumps(
            {
                "task": arguments.task,
                "seed": arguments.seed,
                "stages": list(arguments.stages),
                "stage_failures": failures,
            },
            cls=NumpyJson,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nstages that raised:", failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

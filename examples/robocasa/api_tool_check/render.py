"""Turn ``probe.py``'s raw dump into human-checkable figures and one HTML report.

Run with the matplotlib-bearing venv (the harness venv has none, on purpose)::

    /home/felix/robocasa_sim/cap-x/.venv-client/bin/python \
        examples/robocasa/api_tool_check/render.py

Reads only ``out/raw/``; writes only ``out/figures/``, ``out/index.html`` and
``out/REPORT.md``. The report is organised by TOOL, not by stage, because the question
being answered is "does each of the 31 tools behave?".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import html
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from vis import (
    FIGURES,
    OUT,
    PALETTE,
    RAW,
    SERIES,
    StageData,
    box_segments,
    draw_box,
    draw_segments_2d,
    draw_segments_3d,
    equalize,
    filmstrip,
    frame_segments,
    gripper_segments,
    mask_overlay,
    project,
    save,
    scatter_cloud,
    show_image,
    turntable_gif,
)

TOOL_ORDER = [
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
    "move_to_joints",
    "move_to_pose",
    "move_synchronized",
    "open_gripper",
    "close_gripper",
    "set_gripper",
    "set_grippers",
    "go_home",
    "robocasa.get_task_metadata",
    "robocasa.get_controller_metadata",
    "robocasa.get_base_pose",
    "robocasa.navigate_to_pose",
    "robocasa.base_pose_is_free",
    "robocasa.plan_standoff_pose",
    "robocasa.plan_approach_pose",
]


@dataclass
class Asset:
    tool: str
    title: str
    path: str
    caption: str = ""
    kind: str = "image"


ASSETS: list[Asset] = []


def add(tool: str, title: str, path: str | None, caption: str = "", kind: str = "image") -> None:
    if path:
        ASSETS.append(Asset(tool=tool, title=title, path=path, caption=caption, kind=kind))


def video(tool: str, stage: str, filename: str, title: str, caption: str = "") -> None:
    path = RAW / stage / filename
    if path.exists():
        ASSETS.append(
            Asset(
                tool=tool,
                title=title,
                path=str(path.relative_to(OUT)),
                caption=caption,
                kind="video",
            )
        )


# --------------------------------------------------------------------------- identity


def render_identity() -> None:
    stage = StageData("identity")
    if not stage.exists():
        return
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0))
    for row, key in enumerate(("agentview", "eye_in_hand")):
        rgb = stage.image(f"rgb_{key}")
        depth = stage.get(f"depth_{key}")
        show_image(axes[row][0], rgb, f"{key}: RGB 512x512")
        image = axes[row][1].imshow(depth, cmap="viridis")
        axes[row][1].set_title(f"{key}: depth (m)")
        axes[row][1].set_xticks([])
        axes[row][1].set_yticks([])
        axes[row][1].grid(False)
        fig.colorbar(image, ax=axes[row][1], fraction=0.046)
    fig.suptitle("get_observation(): what the agent actually receives", y=0.995)
    fig.tight_layout()
    add(
        "get_observation",
        "Both cameras: RGB and metric depth",
        save(fig, "identity", "cameras.png"),
        "512x512 RGB plus a float depth map in metres for each camera, with intrinsics and a "
        "camera pose expressed in robot0_link0.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for axis, key in zip(axes, ("agentview", "eye_in_hand"), strict=True):
        residual = stage.get(f"reprojection_residual_{key}")
        axis.hist(np.log10(np.maximum(residual, 1e-18)), bins=60, color=SERIES[0])
        axis.set_xlabel("log10 reprojection error (px)")
        axis.set_ylabel("pixels")
        axis.set_title(f"{key}: max {residual.max():.1e} px")
    fig.suptitle(
        "Calibration round trip: every depth pixel back-projected to robot0_link0 and "
        "reprojected with the published K and extrinsics"
    )
    fig.tight_layout()
    add(
        "mask_to_point_cloud",
        "Calibration round trip is exact",
        save(fig, "identity", "reprojection.png"),
        "Back-projection and reprojection agree to ~1e-13 px, so the published intrinsics and "
        "extrinsics are internally consistent to float precision.",
    )

    # Does the reported end-effector pose land on the gripper you can see?
    ee = stage.get("ee_pose_matrix")
    if ee is not None:
        intrinsics = stage.get("intrinsics_agentview")
        camera_pose = stage.get("cam_pose_agentview")
        rgb = stage.image("rgb_agentview")
        jaw = float(stage.get("gripper_position")[0])
        hand = ee.copy()
        hand[:3, 3] = ee[:3, 3] + ee[:3, 2] * -0.10
        hand[:3, :3] = ee[:3, :3] @ np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        views = [
            ("agentview (scene camera)", rgb, intrinsics, camera_pose, False),
            (
                "robot0_eye_in_hand (camera bolted to this hand)",
                stage.image("rgb_eye_in_hand"),
                stage.get("intrinsics_eye_in_hand"),
                stage.get("cam_pose_eye_in_hand"),
                True,
            ),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8))
        for axis, (title, image, k, pose, draw_hand) in zip(axes, views, strict=True):
            show_image(axis, image, title)
            if draw_hand:
                draw_segments_2d(
                    axis, gripper_segments(hand, 0.08 * jaw), k, pose,
                    color="#00e5ff", linewidth=2.6,
                )
            for start, end, color in frame_segments(ee, 0.06):
                draw_segments_2d(axis, [(start, end)], k, pose, color=color, linewidth=3.0)
            origin = project(ee[:3, 3].reshape(1, 3), k, pose)[0]
            axis.plot(origin[0], origin[1], "o", color="#ffd400", markersize=8,
                      markeredgecolor="black")
            axis.set_xlim(0, image.shape[1])
            axis.set_ylim(image.shape[0], 0)
        fig.suptitle(
            "get_robot_state().end_effector_poses['primary'], projected with the published "
            "calibration\nyellow dot = the reported origin, RGB triad = its x/y/z, cyan = the "
            f"Panda hand it implies at the measured jaw width ({jaw:.2f})"
        )
        fig.tight_layout()
        add(
            "get_robot_state",
            "The reported end-effector pose, drawn on the picture",
            save(fig, "identity", "robot_state_ee.png"),
            "In the wrist view the reported origin lands between the two visible fingertips, "
            "which is where a Panda's gripper0_right_eef frame is defined. In the wrong frame "
            "(robot0_base, the documented decoy) it would be ~14 m away and project nowhere.",
        )

    points = stage.get("scene_cloud_agentview")
    colors = stage.get("scene_colors_agentview")

    def draw(axis: plt.Axes) -> None:
        scatter_cloud(axis, points, colors=colors, size=1.2, alpha=0.75)
        for segment, color in (
            (frame_segments(np.eye(4), 0.15)[0], "#d62728"),
            (frame_segments(np.eye(4), 0.15)[1], "#2ca02c"),
            (frame_segments(np.eye(4), 0.15)[2], "#1f77b4"),
        ):
            axis.plot(
                [segment[0][0], segment[1][0]],
                [segment[0][1], segment[1][1]],
                [segment[0][2], segment[1][2]],
                color=color,
                linewidth=2.5,
            )
        equalize(axis, points)
        axis.set_title("agentview depth -> robot0_link0 (origin triad = the arm base)")

    add(
        "get_observation",
        "The whole scene as a point cloud in robot0_link0",
        turntable_gif(draw, "identity", "scene_cloud.gif"),
        "Every valid depth pixel of the scene camera, coloured by its own RGB. The red/green/"
        "blue triad at the origin is robot0_link0 (x/y/z).",
    )


# --------------------------------------------------------------------------- perception


def render_perception() -> None:
    stage = StageData("perception")
    if not stage.exists():
        return
    records = stage.records_for("segment_text")

    for camera_key, camera_name in (("agentview", "agentview"), ("eye_in_hand", "robot0_eye_in_hand")):
        rgb = stage.image(f"rgb_{camera_key}")
        prompts = [
            record
            for record in records
            if record["inputs"].get("camera_name") == camera_name
            and "text" in record["inputs"]
        ]
        if not prompts:
            continue
        columns = 3
        rows = int(np.ceil(len(prompts) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.6 * rows), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for index, record in enumerate(prompts):
            prompt = record["inputs"]["text"]
            axis = axes[index // columns][index % columns]
            axis.axis("on")
            masks = stage.get(f"mask_{camera_key}_{prompt.replace(' ', '_')}")
            color = PALETTE.get(prompt, SERIES[index % len(SERIES)])
            if masks is None:
                show_image(axis, rgb, f"{prompt!r}: no mask")
                continue
            top = record["output"]["segmentations"][0]
            show_image(
                axis,
                mask_overlay(rgb, masks[0], color),
                f"{prompt!r}  score {top['score']:.2f}  {top['mask_pixels']} px",
            )
            if top.get("box_xyxy"):
                draw_box(axis, top["box_xyxy"], color)
        fig.suptitle(
            f"segment_text({camera_name!r}, prompt) -- best of up to 5 returned masks", y=1.0
        )
        fig.tight_layout()
        add(
            "segment_text",
            f"Text prompts on {camera_name}",
            save(fig, "perception", f"segment_text_{camera_key}.png"),
            "Input: the RGB frame above plus the quoted prompt. Output: the highest-scoring "
            "mask (tinted, outlined) and its box and score.",
        )

    # lower-ranked masks for one prompt, to show what "up to 5" means
    masks = stage.get("mask_agentview_fish")
    if masks is not None:
        rgb = stage.image("rgb_agentview")
        record = next(
            record
            for record in records
            if record["inputs"].get("text") == "fish"
            and record["inputs"]["camera_name"] == "agentview"
        )
        images = [
            (
                f"rank {index}  score {record['output']['segmentations'][index]['score']:.3f}",
                mask_overlay(rgb, masks[index], PALETTE["fish"]),
            )
            for index in range(len(masks))
        ]
        add(
            "segment_text",
            "All ranked masks for one prompt ('fish')",
            filmstrip(images, "perception", "segment_text_ranks.png", columns=3),
            "segment_text returns a ranked list; only index 0 is usually meaningful here.",
        )

    # segment_points
    images = []
    for camera_key, camera_name in (("agentview", "agentview"), ("eye_in_hand", "robot0_eye_in_hand")):
        masks = stage.get(f"pointmask_{camera_key}")
        point = stage.get(f"pointmask_{camera_key}__point")
        if masks is None:
            continue
        rgb = stage.image(f"rgb_{camera_key}")
        fig, axes = plt.subplots(1, len(masks) + 1, figsize=(3.3 * (len(masks) + 1), 3.6))
        show_image(axes[0], rgb, f"{camera_name}\nclick at ({point[0]:.0f}, {point[1]:.0f})")
        axes[0].plot(point[0], point[1], marker="*", markersize=16, color="#ffd400",
                     markeredgecolor="black", markeredgewidth=0.8)
        for index in range(len(masks)):
            show_image(
                axes[index + 1],
                mask_overlay(rgb, masks[index], SERIES[index]),
                f"returned mask {index}",
            )
            axes[index + 1].plot(point[0], point[1], marker="*", markersize=14, color="#ffd400",
                                 markeredgecolor="black", markeredgewidth=0.8)
        fig.suptitle(f"segment_points({camera_name!r}, [[{point[0]:.0f}, {point[1]:.0f}]])")
        fig.tight_layout()
        add(
            "segment_points",
            f"One point prompt on {camera_name}",
            save(fig, "perception", f"segment_points_{camera_key}.png"),
            "Input: a single (x, y) pixel, marked with a star. Output: three candidate masks "
            "(whole object / part / region), ranked by the service's own score.",
        )

    # clouds + OBBs in 3D
    named = [
        name
        for name in ("fish", "plate", "counter")
        if stage.get(f"cloud_{name}") is not None
    ]

    def draw_clouds(axis: plt.Axes) -> None:
        allpoints = []
        for name in named:
            points = stage.get(f"cloud_{name}")
            allpoints.append(points)
            scatter_cloud(
                axis, points, color=PALETTE[name], size=2.0, alpha=0.55, label=name,
                stride=max(1, len(points) // 4000),
            )
            packed = stage.get(f"obb_{name}")
            if packed is not None:
                transform = packed[:16].reshape(4, 4)
                draw_segments_3d(
                    axis, box_segments(transform, packed[16:]), color=PALETTE[name], linewidth=1.4
                )
        equalize(axis, np.vstack(allpoints))
        axis.legend(loc="upper left", fontsize=8)
        axis.set_title("mask_to_point_cloud + estimate_geometry (robot0_link0)")

    if named:
        add(
            "estimate_geometry",
            "Per-object clouds with their estimated oriented bounding boxes",
            turntable_gif(draw_clouds, "perception", "clouds_obb.gif"),
            "Each mask back-projected to robot0_link0, with the PCA box estimate_geometry "
            "returns drawn around it.",
        )

    # OBBs reprojected onto the image -- the easiest way to judge a 3D box by eye
    intrinsics = stage.get("intrinsics_agentview")
    camera_pose = stage.get("cam_pose_agentview")
    rgb = stage.image("rgb_agentview")
    fig, axis = plt.subplots(figsize=(6.6, 6.6))
    show_image(axis, rgb)
    for name in ("fish", "plate", "counter", "pan"):
        packed = stage.get(f"obb_{name}")
        if packed is None:
            continue
        transform = packed[:16].reshape(4, 4)
        draw_segments_2d(
            axis, box_segments(transform, packed[16:]), intrinsics, camera_pose,
            color=PALETTE[name], linewidth=1.8,
        )
        centre = project(transform[:3, 3].reshape(1, 3), intrinsics, camera_pose)[0]
        axis.plot(centre[0], centre[1], "o", color=PALETTE[name], markersize=5)
        offsets = {"fish": -26, "plate": -12, "pan": 2, "counter": 16}
        axis.text(centre[0] + 8, centre[1] + offsets.get(name, 0), name, color="white",
                  fontsize=8, bbox={"facecolor": PALETTE[name], "edgecolor": "none", "pad": 1.2})
    axis.set_xlim(0, rgb.shape[1])
    axis.set_ylim(rgb.shape[0], 0)
    axis.set_title("estimate_geometry boxes projected back onto the camera image")
    add(
        "estimate_geometry",
        "The same boxes drawn back onto the picture",
        save(fig, "perception", "obb_overlay.png"),
        "A 3D box is hard to judge in a scatter plot; projected through the published "
        "calibration it either sits on the object or it does not.",
    )

    # cross-camera agreement
    if stage.get("wristcloud_fish") is not None:

        def draw_cross(axis: plt.Axes) -> None:
            scene = stage.get("cloud_fish")
            wrist = stage.get("wristcloud_fish")
            scatter_cloud(axis, scene, color="#2e86ab", size=4, alpha=0.7, label="from agentview")
            scatter_cloud(axis, wrist, color="#d1495b", size=4, alpha=0.5, label="from wrist")
            equalize(axis, np.vstack([scene, wrist]))
            axis.legend(fontsize=8)
            axis.set_title("same object, two cameras, one frame")

        add(
            "mask_to_point_cloud",
            "Cross-camera agreement on the fish",
            turntable_gif(draw_cross, "perception", "cross_camera.gif"),
            "Independent back-projections from two cameras with different intrinsics and "
            "extrinsics land on top of each other -- the check a self-consistent reprojection "
            "cannot make.",
        )

    # crop
    before = stage.get("crop_counter_in")
    after = stage.get("crop_counter_out")
    box = stage.get("crop_box")
    if before is not None:
        fig, axis = plt.subplots(figsize=(6.0, 5.2))
        axis.scatter(before[::9, 0], before[::9, 1], s=1.5, color="#c7ccd1", label="input (counter)")
        axis.scatter(after[::9, 0], after[::9, 1], s=1.5, color=PALETTE["counter"], label="kept")
        axis.add_patch(
            plt.Rectangle(
                (box[0][0], box[0][1]),
                box[1][0] - box[0][0],
                box[1][1] - box[0][1],
                fill=False,
                edgecolor="#d62728",
                linewidth=1.8,
                label="lower_xyz / upper_xyz",
            )
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
        axis.legend(fontsize=8)
        axis.set_title(f"crop_point_cloud: {len(before)} -> {len(after)} points (top view)")
        add(
            "crop_point_cloud",
            "Axis-aligned crop of the counter cloud",
            save(fig, "perception", "crop.png"),
            "Red rectangle is the requested XY range; only points inside all three ranges "
            "survive.",
        )

    # localize_object
    rgb = stage.image("rgb_agentview")
    queries = [
        record["inputs"]["query"]
        for record in stage.records_for("localize_object")
        if record["ok"]
    ]
    if queries:
        columns = min(4, len(queries))
        rows = int(np.ceil(len(queries) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 3.8 * rows), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for index, query in enumerate(queries):
            key = query.replace(" ", "_")
            mask = stage.get(f"localize_mask_{key}")
            packed = stage.get(f"localize_obb_{key}")
            if mask is None:
                continue
            axis = axes[index // columns][index % columns]
            axis.axis("on")
            color = PALETTE.get(query, SERIES[index % len(SERIES)])
            show_image(axis, mask_overlay(rgb, mask, color))
            transform = packed[:16].reshape(4, 4)
            draw_segments_2d(
                axis, box_segments(transform, packed[16:]), intrinsics, camera_pose, color="white"
            )
            centre = project(transform[:3, 3].reshape(1, 3), intrinsics, camera_pose)[0]
            axis.plot(centre[0], centre[1], "x", color="white", markersize=9, markeredgewidth=2)
            axis.set_title(
                f"{query!r}\ncentre {np.round(transform[:3, 3], 3).tolist()} m\n"
                f"extents {np.round(packed[16:], 3).tolist()} m",
                fontsize=8,
            )
            axis.set_xlim(0, rgb.shape[1])
            axis.set_ylim(rgb.shape[0], 0)
        fig.suptitle("localize_object(query): mask + cloud + box, composed", y=1.0)
        fig.tight_layout()
        add(
            "localize_object",
            "One call per query: mask, cloud and box together",
            save(fig, "perception", "localize.png"),
            "White wireframe is the returned geometry, white cross its centre, reprojected "
            "into the image. The centre is what a program would reach for.",
        )


def _quat_to_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# --------------------------------------------------------------------------- grasps


def _grasp_overlay_figure(
    stage: StageData, name: str, prompt: str, title: str
) -> str | None:
    poses = stage.get(f"grasps_{prompt}")
    if poses is None:
        return None
    meta = stage.get(f"graspmeta_{prompt}")
    contacts = stage.get(f"graspcontact_{prompt}")
    mask = stage.get(f"mask_{prompt}")
    rgb = stage.image("rgb_agentview")
    intrinsics = stage.get("intrinsics_agentview")
    camera_pose = stage.get("cam_pose_agentview")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.3))
    for axis, zoom in zip(axes, (False, True), strict=True):
        show_image(axis, mask_overlay(rgb, mask, PALETTE.get(prompt, "#d1495b"), 0.3))
        for index, transform in enumerate(poses):
            color = SERIES[index % len(SERIES)]
            draw_segments_2d(
                axis, gripper_segments(transform, float(meta[index][1])), intrinsics,
                camera_pose, color=color, linewidth=2.2,
            )
            origin = project(transform[:3, 3].reshape(1, 3), intrinsics, camera_pose)[0]
            axis.plot(origin[0], origin[1], "o", color=color, markersize=4)
            axis.text(origin[0] + 4, origin[1] - 4, f"{meta[index][0]:.2f}", color="white",
                      fontsize=7, bbox={"facecolor": color, "edgecolor": "none", "pad": 1.0})
            if contacts is not None:
                contact = project(contacts[index].reshape(1, 3), intrinsics, camera_pose)[0]
                axis.plot(contact[0], contact[1], "x", color=color, markersize=8,
                          markeredgewidth=1.8)
        if zoom and mask is not None and mask.any():
            rows, columns = np.nonzero(mask)
            pad = 90
            axis.set_xlim(max(0, columns.min() - pad), min(rgb.shape[1], columns.max() + pad))
            axis.set_ylim(min(rgb.shape[0], rows.max() + pad), max(0, rows.min() - pad))
            axis.set_title("zoomed on the target")
        else:
            axis.set_xlim(0, rgb.shape[1])
            axis.set_ylim(rgb.shape[0], 0)
            axis.set_title("full frame")
    fig.suptitle(title)
    fig.tight_layout()
    return save(fig, stage.name, name)


def render_grasps() -> None:
    stage = StageData("grasps")
    if not stage.exists():
        return
    for prompt in ("fish", "pan", "plate"):
        path = _grasp_overlay_figure(
            stage,
            f"grasps_{prompt}.png",
            prompt,
            f"generate_grasps('agentview', segment_text({prompt!r})[0]) -- gripper drawn at "
            "each candidate pose; x marks the reported contact point, the number is the score",
        )
        add(
            "generate_grasps",
            f"Candidates on {prompt!r}",
            path,
            "The drawn gripper is a Panda hand placed at the returned pose with the returned "
            "width, fingertips 0.1034 m along its local +Z.",
        )

        poses = stage.get(f"grasps_{prompt}")
        cloud = stage.get(f"cloud_{prompt}")
        if poses is None or cloud is None:
            continue
        meta = stage.get(f"graspmeta_{prompt}")
        contacts = stage.get(f"graspcontact_{prompt}")

        def draw(axis: plt.Axes, poses=poses, cloud=cloud, meta=meta, contacts=contacts) -> None:
            scatter_cloud(
                axis, cloud, colors=stage.get(f"cloudcolor_{prompt}"), size=4, alpha=0.8,
                stride=max(1, len(cloud) // 3000),
            )
            for index, transform in enumerate(poses):
                color = SERIES[index % len(SERIES)]
                draw_segments_3d(
                    axis, gripper_segments(transform, float(meta[index][1])), color=color,
                    linewidth=2.0, label=f"score {meta[index][0]:.2f}",
                )
                if contacts is not None:
                    axis.scatter(*contacts[index], color=color, marker="x", s=40)
            points = np.vstack([cloud] + [t[:3, 3].reshape(1, 3) for t in poses])
            equalize(axis, points)
            axis.legend(fontsize=8)
            axis.set_title(f"{prompt}: cloud + grasp candidates (robot0_link0)")

        add(
            "generate_grasps",
            f"{prompt!r} candidates in 3D",
            turntable_gif(draw, "grasps", f"grasps_{prompt}_3d.gif"),
            "Same candidates as above, now in 3D against the object's own points.",
        )

    # what select_grasp picked
    rows = []
    for record in stage.records_for("select_grasp"):
        selected = record["output"]["selected"]
        rows.append(
            (
                record["label"],
                "none" if selected is None else f"score {selected['score']:.3f}",
            )
        )
    if rows:
        fig, axis = plt.subplots(figsize=(7.0, 0.5 + 0.42 * len(rows)))
        axis.axis("off")
        table = axis.table(
            cellText=[[label, value] for label, value in rows],
            colLabels=["call", "selection"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        axis.set_title("select_grasp(grasps, strategy)")
        add(
            "select_grasp",
            "What each strategy picked",
            save(fig, "grasps", "select_grasp.png"),
            "'top_down' filters on the approach axis pointing down; on this scene most "
            "candidates approach from the side, so it returns None.",
        )


def render_graspexec() -> None:
    stage = StageData("graspexec")
    if not stage.exists():
        return
    rgb = stage.image("rgb_before")
    intrinsics = stage.get("intrinsics_agentview")
    camera_pose = stage.get("cam_pose_agentview")
    grasp = stage.get("grasp_pose")
    cloud = stage.get("cloud_fish")
    if grasp is None:
        return

    # Where did the hand actually finish, compared with the candidate it was aimed at?
    realized = None
    positions = stage.get("grasp_02_grasp__ee_position")
    quaternions = stage.get("grasp_02_grasp__ee_quaternion_wxyz")
    if positions is not None and len(positions):
        rotation = _quat_to_matrix(quaternions[-1])
        realized = np.eye(4)
        # eef -> panda_hand, the same mapping PyRoki applies when solving
        realized[:3, :3] = rotation @ np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        realized[:3, 3] = positions[-1] + rotation @ np.array([0.0, 0.0, -0.10])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4))
    for axis, zoom in zip(axes, (False, True), strict=True):
        show_image(axis, rgb)
        pixels = project(cloud, intrinsics, camera_pose)
        axis.scatter(pixels[:, 0], pixels[:, 1], s=1.0, color="#ffd400", alpha=0.6)
        entries = [(grasp, "#d62728", "candidate pose (drawn as a Panda hand)")]
        if realized is not None:
            entries.append((realized, "#00b0ff", "hand actually reached, from the measured EE"))
        for transform, color, _ in entries:
            draw_segments_2d(
                axis, gripper_segments(transform), intrinsics, camera_pose, color=color,
                linewidth=2.6,
            )
            origin = project(transform[:3, 3].reshape(1, 3), intrinsics, camera_pose)[0]
            axis.plot(origin[0], origin[1], "o", color=color, markersize=5)
        if zoom:
            axis.set_xlim(60, 300)
            axis.set_ylim(430, 230)
            axis.set_title("zoomed on the fish")
        else:
            axis.set_xlim(0, rgb.shape[1])
            axis.set_ylim(rgb.shape[0], 0)
            axis.set_title("full frame")
        axis.legend(
            handles=[plt.Line2D([], [], color=color, label=label) for _, color, label in entries]
            + [plt.Line2D([], [], color="#ffd400", marker="o", linestyle="", label="fish cloud")],
            loc="lower left",
            fontsize=8,
        )
    gap = (
        None
        if realized is None
        else float(np.linalg.norm(realized[:3, 3] - grasp[:3, 3]))
    )
    fig.suptitle(
        "generate_grasps candidate vs the hand pose actually reached"
        + ("" if gap is None else f"  --  {1000 * gap:.0f} mm apart")
    )
    fig.tight_layout()
    add(
        "generate_grasps",
        "Candidate pose vs the hand pose actually reached",
        save(fig, "graspexec", "grasp_conventions.png"),
        "The candidate (red) is a gripper BASE pose: its fingertips, 0.1034 m along its local "
        "+Z, are what touch the object. Commanding it as a gripper0_right_eef target only "
        "works after adding that offset back; the blue hand is where the arm finished, "
        "reconstructed from the measured end-effector pose.",
    )

    images = []
    for name, caption in (
        ("grasp_01_standoff_scene", "1. at the standoff"),
        ("grasp_02_grasp_scene", "2. at the grasp pose"),
        ("grasp_04_lift_scene", "4. after the 12 cm lift"),
    ):
        path = stage.directory / f"{name}.png"
        if path.exists():
            images.append((caption, stage.image(name)))
    for name, caption in (
        ("grasp_02_grasp_wrist", "2. wrist, at the grasp pose"),
        ("grasp_03_close_wrist", "3. wrist, after close_gripper"),
        ("grasp_04_lift_wrist", "4. wrist, after the lift"),
    ):
        path = stage.directory / f"{name}.png"
        if path.exists():
            images.append((caption, stage.image(name)))
    if images:
        add(
            "generate_grasps",
            "Driving the best candidate: approach, close, lift",
            filmstrip(images, "graspexec", "sequence.png", columns=3),
            "The arm reaches the candidate to within 9 mm, but the jaws close beside the fish "
            "and the lift leaves it on the plate.",
        )

    series = [
        ("grasp_01_standoff", "approach to standoff"),
        ("grasp_02_grasp", "descend to grasp"),
        ("grasp_03_close", "close"),
        ("grasp_04_lift", "lift"),
    ]
    paths = [
        (label, stage.get(f"{key}__ee_position"))
        for key, label in series
        if stage.get(f"{key}__ee_position") is not None
    ]
    if paths and cloud is not None:

        def draw(axis: plt.Axes) -> None:
            scatter_cloud(axis, cloud, color=PALETTE["fish"], size=5, alpha=0.6, label="fish")
            for index, (label, points) in enumerate(paths):
                axis.plot(points[:, 0], points[:, 1], points[:, 2], color=SERIES[index],
                          linewidth=2.0, label=label)
            draw_segments_3d(axis, gripper_segments(grasp), color="#d62728", linewidth=2.0,
                             label="grasp candidate")
            everything = np.vstack([cloud] + [points for _, points in paths])
            equalize(axis, everything)
            axis.legend(fontsize=7)
            axis.set_title("end-effector path through the grasp attempt")

        add(
            "move_to_pose",
            "End-effector path of the whole grasp attempt",
            turntable_gif(draw, "graspexec", "grasp_path_3d.gif"),
            "Measured gripper0_right_eef position at every control tick, against the fish "
            "cloud and the candidate pose.",
        )

    for key, label in series:
        video("move_to_pose", "graspexec", f"{key}__agentview.mp4", f"Scene camera: {label}")
        video(
            "move_to_pose",
            "graspexec",
            f"{key}__robot0_eye_in_hand.mp4",
            f"Wrist camera: {label}",
        )


# --------------------------------------------------------------------------- ik


def render_ik() -> None:
    stage = StageData("ik")
    if not stage.exists():
        return
    sweep = stage.get("reach_sweep")
    if sweep is not None:
        fig, axis = plt.subplots(figsize=(6.6, 3.2))
        for code, label, color in ((0.0, "+z (up)", SERIES[1]), (1.0, "+x (forward)", SERIES[0])):
            rows = sweep[sweep[:, 0] == code]
            axis.plot(rows[:, 1] * 100, rows[:, 2], "o-", color=color, label=label)
        axis.set_xlabel("displacement from the home end-effector pose (cm)")
        axis.set_ylabel("solve_ik ok")
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["IK_FAILED", "solved"])
        axis.legend()
        axis.set_title("solve_ik reach sweep from the spawn pose (orientation held fixed)")
        add(
            "solve_ik",
            "Where IK stops solving",
            save(fig, "ik", "reach_sweep.png"),
            "Upward motion from the spawn pose fails beyond 2 cm; forward motion survives to "
            "12 cm. The arm starts near the top of its workspace.",
        )

    verifications = []
    for note in stage.notes:
        try:
            verifications = json.loads(note)["ik_execution_verification"]
        except (ValueError, KeyError):
            continue
    if verifications:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
        labels = [row["label"] for row in verifications]
        axes[0].bar(labels, [1000 * row["position_error_m"] for row in verifications],
                    color=SERIES[0])
        axes[0].set_ylabel("position error (mm)")
        axes[0].set_title("where the EE landed vs the pose IK was given")
        axes[1].bar(labels, [row["orientation_error_deg"] for row in verifications],
                    color=SERIES[1])
        axes[1].set_ylabel("orientation error (deg)")
        axes[1].set_title("orientation error after driving the solution")
        fig.tight_layout()
        add(
            "solve_ik",
            "IK verified by actually driving the solution",
            save(fig, "ik", "verification.png"),
            "Each IK solution was commanded with move_to_joints; the bars are the gap between "
            "the pose requested and the pose measured afterwards.",
        )

    for key, label in (("verify_mz_6cm", "-z 6 cm"), ("verify_py_6cm", "+y 6 cm")):
        points = stage.get(f"{key}__ee_position")
        if points is None:
            continue
        home = stage.get("home_ee")
        target = home[:3, 3] + (
            np.array([0, 0, -0.06]) if key == "verify_mz_6cm" else np.array([0, 0.06, 0])
        )

        def draw(axis: plt.Axes, points=points, target=target, home=home, label=label) -> None:
            axis.plot(points[:, 0], points[:, 1], points[:, 2], "-o", color=SERIES[0],
                      markersize=2.5, label="measured EE path")
            axis.scatter(*home[:3, 3], color="#999999", s=40, label="start")
            axis.scatter(*target, color="#d62728", s=60, marker="*", label="pose given to IK")
            equalize(axis, np.vstack([points, target.reshape(1, 3), home[:3, 3].reshape(1, 3)]),
                     pad=0.02)
            axis.legend(fontsize=8)
            axis.set_title(f"solve_ik({label}) -> move_to_joints")

        add(
            "solve_ik",
            f"Path to the IK solution ({label})",
            turntable_gif(draw, "ik", f"{key}_3d.gif"),
            "The star is the Cartesian pose handed to solve_ik; the line is where the arm "
            "actually went when its joint answer was commanded.",
        )
        video("solve_ik", "ik", f"{key}__agentview.mp4", f"Scene camera: driving the IK solution ({label})")


# --------------------------------------------------------------------------- planning


def render_planning() -> None:
    stage = StageData("planning")
    if not stage.exists():
        return
    plans = {
        key: value for key, value in stage.arrays.items() if key.startswith("plan_")
    }
    if plans:
        fig, axes = plt.subplots(1, len(plans), figsize=(4.0 * len(plans), 3.4), squeeze=False)
        for index, (key, waypoints) in enumerate(sorted(plans.items())):
            axis = axes[0][index]
            for joint in range(waypoints.shape[1]):
                axis.plot(waypoints[:, joint], color=SERIES[joint % len(SERIES)], linewidth=1.3,
                          label=f"j{joint}" if index == 0 else None)
            axis.set_xlabel("waypoint")
            axis.set_ylabel("rad" if index == 0 else "")
            axis.set_title(key.replace("plan_", "").replace("__", "\n"), fontsize=8)
        axes[0][0].legend(fontsize=6, ncol=2)
        fig.suptitle("plan_motion: the joint path each strategy returns for the same goal")
        fig.tight_layout()
        add(
            "plan_motion",
            "Joint paths from each planner",
            save(fig, "planning", "plans.png"),
            "Interpolation returns a 3-point straight line in joint space; cuRobo returns a "
            "smooth collision-aware path of 27-52 waypoints.",
        )

    commanded = stage.get("execute_curobo_plan__commanded_joints")
    measured = stage.get("execute_curobo_plan__measured_joints")
    if commanded is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
        for joint in range(commanded.shape[1]):
            axes[0].plot(commanded[:, joint], "--", color=SERIES[joint % len(SERIES)],
                         linewidth=1.0)
            axes[0].plot(measured[:, joint], "-", color=SERIES[joint % len(SERIES)],
                         linewidth=1.4, label=f"j{joint}")
        axes[0].set_title("commanded (dashed) vs measured (solid) joints")
        axes[0].set_xlabel("control tick")
        axes[0].set_ylabel("rad")
        axes[0].legend(fontsize=6, ncol=2)
        error = np.abs(measured - commanded)
        axes[1].plot(error.max(axis=1), color="#d62728")
        axes[1].set_title("max per-tick tracking error")
        axes[1].set_xlabel("control tick")
        axes[1].set_ylabel("rad")
        fig.tight_layout()
        add(
            "execute_trajectory",
            "Tracking of the cuRobo plan",
            save(fig, "planning", "execute_tracking.png"),
            "The arm takes absolute joint targets, so it lags each waypoint by roughly one "
            "tick and converges when the goal is held.",
        )
        points = stage.get("execute_curobo_plan__ee_position")

        def draw(axis: plt.Axes) -> None:
            axis.plot(points[:, 0], points[:, 1], points[:, 2], "-o", color=SERIES[2],
                      markersize=2.5)
            axis.scatter(*points[0], color="#999999", s=40, label="start")
            axis.scatter(*points[-1], color="#d62728", s=40, label="end")
            equalize(axis, points, pad=0.02)
            axis.legend(fontsize=8)
            axis.set_title("end-effector path while executing the cuRobo plan")

        add(
            "execute_trajectory",
            "End-effector path of the executed plan",
            turntable_gif(draw, "planning", "execute_path_3d.gif"),
        )
    video("execute_trajectory", "planning", "execute_curobo_plan__agentview.mp4",
          "Scene camera: executing the cuRobo plan")
    video("execute_trajectory", "planning", "execute_curobo_plan__robot0_eye_in_hand.mp4",
          "Wrist camera: executing the cuRobo plan")


# --------------------------------------------------------------------------- gripper


def render_gripper() -> None:
    stage = StageData("gripper")
    if not stage.exists():
        return
    order = [
        ("gripper_open_gripper_0.99", "open_gripper()"),
        ("gripper_close_gripper_0.01", "close_gripper()"),
        ("gripper_set_gripper_0.98", "set_gripper(1.0)"),
        ("gripper_set_gripper_0.01", "set_gripper(0.0)"),
        ("gripper_set_grippers_0.98", "set_grippers({'primary': 1.0})"),
    ]
    fig, axis = plt.subplots(figsize=(9.0, 3.2))
    offset = 0
    for index, (key, label) in enumerate(order):
        measured = stage.get(f"{key}__gripper")
        commanded = stage.get(f"{key}__commanded_gripper")
        if measured is None:
            continue
        ticks = np.arange(offset, offset + len(measured))
        axis.plot(ticks, measured, color=SERIES[index % len(SERIES)], linewidth=1.8)
        axis.plot(ticks, commanded, "--", color=SERIES[index % len(SERIES)], linewidth=1.0)
        axis.axvspan(offset, offset + len(measured), color=SERIES[index % len(SERIES)], alpha=0.07)
        axis.text(offset + 1, 1.06, label, fontsize=7, rotation=0)
        offset += len(measured)
    axis.set_ylim(-0.05, 1.2)
    axis.set_xlabel("control tick")
    axis.set_ylabel("normalised jaw (1 = open)")
    axis.set_title("measured jaw (solid) vs commanded (dashed) across every gripper call")
    path = save(fig, "gripper", "gripper_trace.png")
    for tool in ("open_gripper", "close_gripper", "set_gripper", "set_grippers"):
        add(
            tool,
            "Jaw width through every gripper call",
            path,
            "Each call runs a fixed 30-tick settle loop; the jaw reaches 0.98 open / 0.01 "
            "closed and holds. set_gripper(0.5) is refused before any tick is spent.",
        )


# --------------------------------------------------------------------------- motion


def render_motion() -> None:
    stage = StageData("motion")
    if not stage.exists():
        return

    def tracking(key: str, title: str, target: np.ndarray | None = None) -> str | None:
        commanded = stage.get(f"{key}__commanded_joints")
        measured = stage.get(f"{key}__measured_joints")
        if commanded is None:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
        for joint in range(measured.shape[1]):
            axes[0].plot(commanded[:, joint], "--", color=SERIES[joint % len(SERIES)],
                         linewidth=1.0)
            axes[0].plot(measured[:, joint], "-", color=SERIES[joint % len(SERIES)],
                         linewidth=1.4, label=f"j{joint}")
        if target is not None:
            for joint in range(len(target)):
                axes[0].axhline(target[joint], color=SERIES[joint % len(SERIES)], alpha=0.25,
                                linewidth=0.8)
        axes[0].set_xlabel("control tick")
        axes[0].set_ylabel("rad")
        axes[0].set_title("commanded (dashed) vs measured (solid)")
        axes[0].legend(fontsize=6, ncol=2)
        axes[1].plot(np.abs(measured - commanded).max(axis=1), color="#d62728")
        axes[1].set_xlabel("control tick")
        axes[1].set_ylabel("rad")
        axes[1].set_title("max tracking error")
        fig.suptitle(title)
        fig.tight_layout()
        return save(fig, "motion", f"{key}_tracking.png")

    add(
        "move_to_joints",
        "Servo to an absolute joint target",
        tracking("move_to_joints", "move_to_joints(target, tolerance=0.01, max_steps=120)",
                 stage.get("move_to_joints__target")),
        "Horizontal lines are the requested joint angles; the loop re-commands the same "
        "absolute target every tick until the measurement is inside tolerance.",
    )
    add(
        "execute_trajectory",
        "Hand-built 25-waypoint ramp",
        tracking("execute_trajectory", "execute_trajectory(Trajectory(25 waypoints, dt=0.05 s))"),
        "The adapter densifies to 0.05 rad per tick and then servos the final waypoint, which "
        "is why the tick count exceeds the waypoint count.",
    )
    add("go_home", "go_home()", tracking("go_home", "go_home()", stage.get("home_joints")),
        "Returns to the joint vector captured when the API was constructed.")
    video("move_to_joints", "motion", "move_to_joints__agentview.mp4", "Scene camera: move_to_joints")
    video("execute_trajectory", "motion", "execute_trajectory__agentview.mp4",
          "Scene camera: execute_trajectory")
    video("go_home", "motion", "go_home__agentview.mp4", "Scene camera: go_home")

    # move_to_pose: commanded vs measured displacement
    records = [
        record
        for record in stage.records_for("move_to_pose")
        if "commanded_delta_m" in record["output"] and record["output"]["execution"]["ok"]
    ]
    if records:
        labels = [record["label"].split("(")[1].rstrip(")") for record in records]
        commanded = np.asarray([record["output"]["commanded_delta_m"] for record in records])
        measured = np.asarray([record["output"]["measured_delta_m"] for record in records])
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
        width = 0.35
        index = np.arange(len(labels))
        for axis_index, axis in enumerate(axes):
            if axis_index == 0:
                axis.bar(index - width / 2, 1000 * np.linalg.norm(commanded, axis=1),
                         width, label="commanded |delta|", color="#c7ccd1")
                axis.bar(index + width / 2, 1000 * np.linalg.norm(measured, axis=1),
                         width, label="measured |delta|", color=SERIES[0])
                axis.set_ylabel("mm")
                axis.set_title(
                    "displacement asked for vs achieved\n(refused/unreachable calls excluded)"
                )
            else:
                errors = [1000 * record["output"]["position_error_m"] for record in records]
                axis.bar(index, errors, color=SERIES[1])
                axis.set_ylabel("mm")
                axis.set_title("final distance to the requested pose")
            axis.set_xticks(index)
            axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
            axis.legend(fontsize=8) if axis_index == 0 else None
        fig.tight_layout()
        add(
            "move_to_pose",
            "Commanded vs achieved Cartesian displacement",
            save(fig, "motion", "move_to_pose_errors.png"),
            "Each move holds orientation and shifts one axis by 5 cm. Final error stays under "
            "6 mm on every axis tested.",
        )

    keys = [
        key[: -len("__ee_position")]
        for key in stage.arrays
        if key.startswith("move_to_pose_") and key.endswith("__ee_position")
    ]
    if keys:
        home = stage.get("move_to_pose__home_ee")

        def draw(axis: plt.Axes) -> None:
            everything = []
            for index, key in enumerate(sorted(keys)):
                points = stage.get(f"{key}__ee_position")
                everything.append(points)
                axis.plot(points[:, 0], points[:, 1], points[:, 2], "-", color=SERIES[index % 7],
                          linewidth=1.8, label=key.replace("move_to_pose_", ""))
                axis.scatter(*points[-1], color=SERIES[index % 7], s=25)
            axis.scatter(*home[:3, 3], color="black", s=50, marker="*", label="home EE")
            equalize(axis, np.vstack(everything), pad=0.02)
            axis.legend(fontsize=7)
            axis.set_title("move_to_pose: measured end-effector paths")

        add(
            "move_to_pose",
            "Every move_to_pose path in one 3D view",
            turntable_gif(draw, "motion", "move_to_pose_3d.gif"),
            "Measured gripper0_right_eef position at every tick for each requested pose.",
        )
    for key in sorted(keys):
        video("move_to_pose", "motion", f"{key}__agentview.mp4",
              f"Scene camera: {key.replace('move_to_pose_', 'move_to_pose ')}")


# --------------------------------------------------------------------------- base


def render_base() -> None:
    stage = StageData("base")
    if not stage.exists():
        return
    grid = stage.get("free_grid")
    axis_values = stage.get("free_grid_axis")
    spawn = stage.get("spawn_pose")
    counter_world = stage.get("counter_cloud_world")
    track = stage.get("base_track")
    if grid is None:
        return
    extent = [
        spawn[0, 3] + axis_values[0] - 0.05,
        spawn[0, 3] + axis_values[-1] + 0.05,
        spawn[1, 3] + axis_values[0] - 0.05,
        spawn[1, 3] + axis_values[-1] + 0.05,
    ]
    fig, axis = plt.subplots(figsize=(7.2, 6.8))
    axis.imshow(
        grid.astype(float), origin="lower", extent=extent, cmap="RdYlGn", vmin=0, vmax=1,
        alpha=0.85, interpolation="nearest",
    )
    if counter_world is not None:
        axis.scatter(counter_world[::50, 0], counter_world[::50, 1], s=1.0, color="#222222",
                     alpha=0.5, label="counter cloud (robot0_link0 -> world)")
    axis.plot(spawn[0, 3], spawn[1, 3], marker="*", markersize=18, color="#1f77b4",
              markeredgecolor="white", label="spawn (robocasa.get_base_pose)")
    yaw = np.arctan2(spawn[1, 0], spawn[0, 0])
    axis.arrow(spawn[0, 3], spawn[1, 3], 0.25 * np.cos(yaw), 0.25 * np.sin(yaw),
               width=0.015, color="#1f77b4")
    for name, color in (("standoff_0.3", "#8a4fff"), ("standoff_0.6", "#9d4f8c"),
                        ("approach_0.7", "#e0a458"), ("approach_0.4", "#d1495b")):
        value = stage.get(name)
        if value is None:
            continue
        world = spawn @ np.array([value[0], value[1], 0.0, 1.0])
        axis.plot(world[0], world[1], "P", markersize=11, color=color,
                  markeredgecolor="white", label=f"{name} -> world")
    if track is not None:
        axis.plot(track[:, 0], track[:, 1], "-o", color="#111111", markersize=4,
                  linewidth=1.2, label="base pose after each navigate_to_pose")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_aspect("equal")
    axis.legend(fontsize=7, loc="upper left")
    axis.set_title(
        "robocasa.base_pose_is_free, 961 calls on a 0.1 m grid\n"
        "green = footprint fits, red = blocked (yaw held at the spawn heading)"
    )
    path = save(fig, "base", "occupancy.png")
    add(
        "robocasa.base_pose_is_free",
        "An occupancy map built from the yes/no answer",
        path,
        "The tool answers one pose at a time; sweeping it reconstructs the free space. The "
        "black counter points -- pushed into world with robocasa.get_base_pose -- land on the "
        "red band, so the two frames agree.",
    )
    add("robocasa.get_base_pose", "Spawn pose on the occupancy map", path)
    add("robocasa.plan_standoff_pose", "Where the standoff poses land", path)
    add("robocasa.plan_approach_pose", "Where the approach poses land", path)

    records = stage.records_for("robocasa.navigate_to_pose")
    if records:
        labels = [record["label"].split("(")[1].rstrip(")") for record in records]
        commanded = np.asarray([record["output"]["commanded_local"] for record in records])
        measured = np.asarray([record["output"]["measured_local"] for record in records])
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
        for index, (name, unit) in enumerate(
            ((" dx", "m"), (" dy", "m"), (" dyaw", "rad"))
        ):
            axis = axes[index]
            position = np.arange(len(labels))
            axis.bar(position - 0.2, commanded[:, index], 0.4, label="commanded", color="#c7ccd1")
            axis.bar(position + 0.2, measured[:, index], 0.4, label="measured", color=SERIES[index])
            axis.set_xticks(position)
            axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
            axis.set_ylabel(unit)
            axis.set_title(f"{name.strip()} (base frame)")
            if index == 0:
                axis.legend(fontsize=8)
        fig.suptitle("robocasa.navigate_to_pose: relative command vs measured base motion")
        fig.tight_layout()
        add(
            "robocasa.navigate_to_pose",
            "Commanded vs measured base motion",
            save(fig, "base", "navigate.png"),
            "Lateral and yaw track within ~6%. The forward command is blocked by the counter "
            "and stops after 5 cm, which the tool reports as a non-converged ExecutionResult.",
        )
    for path in stage.videos():
        video("robocasa.navigate_to_pose", "base", path.name, f"Scene camera: {path.stem}")


# --------------------------------------------------------------------------- report


def collect_records() -> dict[str, list[tuple[str, dict[str, Any]]]]:
    by_tool: dict[str, list[tuple[str, dict[str, Any]]]] = {name: [] for name in TOOL_ORDER}
    for directory in sorted(RAW.iterdir()):
        if not (directory / "records.json").exists():
            continue
        payload = json.loads((directory / "records.json").read_text(encoding="utf-8"))
        for record in payload["records"]:
            by_tool.setdefault(record["tool"], []).append((directory.name, record))
    return by_tool


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _json_block(value: Any, limit: int = 2600) -> str:
    text = json.dumps(value, indent=1, default=str)
    if len(text) > limit:
        text = text[:limit] + "\n  ... (truncated; full value in out/raw/*/records.json)"
    return f"<pre>{_escape(text)}</pre>"


def build_html(by_tool: dict[str, list[tuple[str, dict[str, Any]]]], summary: str) -> None:
    assets_by_tool: dict[str, list[Asset]] = {}
    for asset in ASSETS:
        assets_by_tool.setdefault(asset.tool, []).append(asset)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>RoboCasa CaP API tool check</title>",
        """<style>
 body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1b1f23}
 .wrap{max-width:1180px;margin:0 auto;padding:24px}
 h1{font-size:26px;margin:0 0 4px} h2{margin:34px 0 6px;font-size:20px;border-top:2px solid #d9dde2;padding-top:14px}
 h3{margin:18px 0 6px;font-size:15px;color:#3b434b}
 code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
 pre{background:#0f1419;color:#e6edf3;padding:10px 12px;border-radius:6px;overflow-x:auto;font-size:11.5px;line-height:1.45}
 table{border-collapse:collapse;width:100%;margin:8px 0;background:#fff}
 th,td{border:1px solid #dfe3e8;padding:5px 8px;text-align:left;vertical-align:top;font-size:12.5px}
 th{background:#eef1f4}
 .ok{color:#116329;font-weight:600}.bad{color:#a40e26;font-weight:600}
 .badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-left:8px}
 .b-ok{background:#d6f5de;color:#116329}.b-part{background:#fdf2c9;color:#7a5c00}.b-bad{background:#ffe0e4;color:#a40e26}
 figure{margin:14px 0;background:#fff;border:1px solid #e3e7ea;border-radius:8px;padding:10px}
 figure img,figure video{max-width:100%;border-radius:4px;display:block}
 figcaption{font-size:12px;color:#5a636c;margin-top:6px}
 .vid{display:inline-block;width:320px;margin:6px 10px 6px 0;vertical-align:top}
 .vid video{width:320px}
 .vid .t{font-size:11.5px;color:#5a636c}
 nav a{display:inline-block;margin:2px 10px 2px 0;font-size:12.5px}
 .sum{background:#fff;border:1px solid #e3e7ea;border-radius:8px;padding:14px 18px}
</style></head><body><div class='wrap'>""",
        "<h1>RoboCasa CaP API tool check</h1>",
        f"<div class='sum'>{summary}</div>",
        "<h2 id='toc'>Tools</h2><nav>",
    ]
    for name in TOOL_ORDER:
        parts.append(f"<a href='#{_escape(name)}'>{_escape(name)}</a>")
    parts.append("</nav>")

    for name in TOOL_ORDER:
        records = by_tool.get(name, [])
        total = len(records)
        passed = sum(1 for _, record in records if record["ok"])
        if total == 0:
            badge = "<span class='badge b-bad'>not exercised</span>"
        elif passed == total:
            badge = f"<span class='badge b-ok'>{passed}/{total} checks behaved</span>"
        elif passed == 0:
            badge = f"<span class='badge b-bad'>0/{total} checks behaved</span>"
        else:
            badge = f"<span class='badge b-part'>{passed}/{total} checks behaved</span>"
        parts.append(f"<h2 id='{_escape(name)}'><code>{_escape(name)}</code>{badge}</h2>")
        if records:
            parts.append(
                "<table><tr><th>call</th><th>input</th><th>result</th><th>ok</th></tr>"
            )
            for stage_name, record in records:
                verdict = record["verdict"] or ""
                parts.append(
                    "<tr><td>"
                    f"<b>{_escape(record['label'])}</b><br><span style='color:#8a9199'>"
                    f"stage {_escape(stage_name)}</span></td>"
                    f"<td>{_json_block(record['inputs'], 900)}</td>"
                    f"<td>{_escape(verdict)}{_json_block(record['output'], 1500)}</td>"
                    f"<td class='{'ok' if record['ok'] else 'bad'}'>"
                    f"{'yes' if record['ok'] else 'NO'}</td></tr>"
                )
            parts.append("</table>")
        images = [a for a in assets_by_tool.get(name, []) if a.kind == "image"]
        videos = [a for a in assets_by_tool.get(name, []) if a.kind == "video"]
        seen = set()
        for asset in images:
            if asset.path in seen:
                continue
            seen.add(asset.path)
            parts.append(
                f"<figure><h3>{_escape(asset.title)}</h3>"
                f"<img src='{_escape(asset.path)}' loading='lazy'>"
                f"<figcaption>{_escape(asset.caption)}</figcaption></figure>"
            )
        if videos:
            parts.append("<figure><h3>Recorded episodes</h3>")
            for asset in videos:
                parts.append(
                    f"<div class='vid'><video src='{_escape(asset.path)}' controls muted "
                    f"loop preload='metadata'></video>"
                    f"<div class='t'>{_escape(asset.title)}</div></div>"
                )
            parts.append("</figure>")

    parts.append("</div></body></html>")
    (OUT / "index.html").write_text("\n".join(parts), encoding="utf-8")


def build_markdown(by_tool: dict[str, list[tuple[str, dict[str, Any]]]]) -> None:
    lines = ["# RoboCasa CaP API tool check -- results table", ""]
    lines.append("| tool | checks | behaved | notes |")
    lines.append("| --- | ---: | ---: | --- |")
    for name in TOOL_ORDER:
        records = by_tool.get(name, [])
        failures = [record["label"] for _, record in records if not record["ok"]]
        lines.append(
            f"| `{name}` | {len(records)} | {len(records) - len(failures)} | "
            + (", ".join(failures) if failures else "-")
            + " |"
        )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for renderer in (
        render_identity,
        render_perception,
        render_grasps,
        render_graspexec,
        render_ik,
        render_planning,
        render_gripper,
        render_motion,
        render_base,
    ):
        print(f"rendering {renderer.__name__} ...")
        renderer()
    by_tool = collect_records()
    run = json.loads((OUT / "probe-run.json").read_text(encoding="utf-8"))
    total = sum(len(records) for records in by_tool.values())
    failed = sum(
        1 for records in by_tool.values() for _, record in records if not record["ok"]
    )
    summary_path = OUT / "SUMMARY.html"
    summary = (
        summary_path.read_text(encoding="utf-8")
        if summary_path.exists()
        else (
            f"<p>{total} checks across {len(TOOL_ORDER)} public tools; {failed} did not "
            f"behave as documented. Task <code>{run['task']}</code>, seed "
            f"<code>{run['seed']}</code>.</p>"
        )
    )
    build_html(by_tool, summary)
    build_markdown(by_tool)
    print(f"\n{total} records, {failed} not-as-expected")
    print(f"wrote {OUT / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The Stretch 4 as seen through the bridge: state, cameras and kinematics.

Plays the role ``RealYamEnv`` plays for YAM: the adapter, the live session
wrapper (``camera_aliases`` / ``camera_frame``) and the MCP server read it.
It never commands motion itself; the adapter does, through ``client``.

Cameras: ``wrist`` is the only calibrated RGB-D camera. Its pose in odom is
forward kinematics of the joint state the bridge paired with that frame. The
three head cameras are view-only: they appear in ``camera_frame`` (the images
``observe`` returns) but not in the CAP ``Observation``, because they have no
depth and no calibrated pose.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .client import DEFAULT_URL, BridgeClient
from .kinematics import JOINT_NAMES, MODEL_PATH, Stretch4Kinematics

STATION = "stretch4"
BASE_FRAME = "odom"
WRIST_CAMERA = "wrist"
CAMERAS = (WRIST_CAMERA, "head_center", "head_left", "head_right")
#: Rate ASPIRE's interpolation planner assumes. The executor runs at 30 Hz;
#: only the final waypoint of each trajectory is sent (see adapter).
CONTROL_FREQUENCY_HZ = 30.0


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    timestamp_s: float  # time.monotonic() at capture, as the live adapter expects


@dataclass(frozen=True)
class WristFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    odom_from_camera: np.ndarray
    timestamp_s: float


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


class Stretch4Station:
    def __init__(self, client: BridgeClient, kinematics: Stretch4Kinematics | None = None) -> None:
        self.client = client
        self.kinematics = kinematics or Stretch4Kinematics()
        self.camera_aliases = CAMERAS
        self.config = SimpleNamespace(
            station=STATION,
            base_frame=BASE_FRAME,
            model_xml=Path(MODEL_PATH),
            camera=SimpleNamespace(role=WRIST_CAMERA),
            controller=SimpleNamespace(control_frequency_hz=CONTROL_FREQUENCY_HZ),
        )

    def joint_state(self) -> tuple[np.ndarray, float, dict]:
        """``(joints[8], gripper 0-1, raw bridge reply)``."""
        state = self.client.state()
        joints = np.array([float(state["joints"][name]) for name in JOINT_NAMES])
        return joints, float(state["gripper"]), state

    def camera_frame(self, name: str) -> CameraFrame | None:
        try:
            frame = self.client.camera(name)
        except Exception:
            return None
        rgb = np.asarray(_decode(frame["jpeg_b64"]).convert("RGB"), dtype=np.uint8)
        return CameraFrame(rgb=rgb, timestamp_s=time.monotonic() - float(frame["age_s"]))

    def wrist_frame(self) -> WristFrame:
        frame = self.client.camera(WRIST_CAMERA)
        rgb = np.asarray(_decode(frame["jpeg_b64"]).convert("RGB"), dtype=np.uint8)
        depth_image = _decode(frame["depth_png_b64"])
        if depth_image.size != (rgb.shape[1], rgb.shape[0]):
            depth_image = depth_image.resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
        depth_m = np.asarray(depth_image, dtype=np.float64) / 1000.0
        joints = np.array([float(frame["joints"][name]) for name in JOINT_NAMES])
        return WristFrame(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=np.asarray(frame["intrinsics"], dtype=np.float64).reshape(3, 3),
            odom_from_camera=self.kinematics.camera_from_joints(joints),
            timestamp_s=time.monotonic() - float(frame["age_s"]),
        )

    def close(self) -> None:
        self.client.close()


def build_station(bridge_url: str = DEFAULT_URL) -> Stretch4Station:
    """Connect to a running bridge; fails fast if it is not there."""
    client = BridgeClient(bridge_url)
    client.health()
    return Stretch4Station(client)


__all__ = ["BASE_FRAME", "CAMERAS", "STATION", "Stretch4Station", "build_station"]

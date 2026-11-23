"""In-process stand-in for ``aspire_bridge``: a kinematic Stretch and synthetic images.

Same HTTP contract as :mod:`cap_harness.stretch4.client`. Moves land instantly
and report ``completed``; targets outside the model's joint limits are
``rejected`` without moving, like the executor. Used by tests and by
``start_system2.sh --station stretch4 --sim``.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException
import numpy as np
from PIL import Image
import uvicorn

from .kinematics import JOINT_NAMES, Stretch4Kinematics

#: The robot's ``kitchen_manipulation`` Home pose (lift, arm and wrists only; Home
#: never moves the base).
FAKE_HOME = {"lift": 0.9, "arm": 0.1, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0}
WRIST_SIZE = (400, 640)  # rows, columns; the 640x400 wrist setting
HEAD_SIZE = (800, 1280)
WRIST_INTRINSICS = [[450.0, 0.0, 320.0], [0.0, 450.0, 200.0], [0.0, 0.0, 1.0]]
FAKE_DEPTH_MM = 500
CAMERAS = ("wrist", "head_center", "head_left", "head_right")


def _jpeg(rows: int, columns: int, tint: int) -> str:
    y, x = np.mgrid[0:rows, 0:columns]
    rgb = np.stack([x * 255 // columns, y * 255 // rows, np.full_like(x, tint)], axis=-1).astype(
        np.uint8
    )
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _depth_png(rows: int, columns: int) -> str:
    depth = np.full((rows, columns), FAKE_DEPTH_MM, dtype=np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(depth).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeRobot:
    """Joint state plus a log of every motion call it received."""

    def __init__(self) -> None:
        kinematics = Stretch4Kinematics()
        self.lower = dict(zip(JOINT_NAMES, kinematics.lower))
        self.upper = dict(zip(JOINT_NAMES, kinematics.upper))
        self.joints = dict.fromkeys(JOINT_NAMES, 0.0) | {"lift": 0.6, "arm": 0.05}
        self.gripper = 0.5
        self.calls: list[tuple[str, dict]] = []
        self.lock = threading.Lock()
        self._images = {
            name: _jpeg(*(WRIST_SIZE if name == "wrist" else HEAD_SIZE), tint=40 * index)
            for index, name in enumerate(CAMERAS)
        }
        self._depth = _depth_png(*WRIST_SIZE)

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "joints": dict(self.joints),
                "gripper": self.gripper,
                "gripper_pct": self.gripper * 90.0,
                "stamp_s": time.time(),
                "age_s": 0.0,
            }

    def camera(self, name: str) -> dict[str, Any]:
        frame = {"jpeg_b64": self._images[name], "stamp_s": time.time(), "age_s": 0.0}
        if name == "wrist":
            with self.lock:
                frame |= {
                    "depth_png_b64": self._depth,
                    "intrinsics": WRIST_INTRINSICS,
                    "joints": dict(self.joints),
                }
        return frame

    def move(self, targets: dict[str, float]) -> dict[str, Any]:
        with self.lock:
            self.calls.append(("move", dict(targets)))
            for name, value in targets.items():
                if name not in self.joints:
                    return {"status": "rejected", "reason": f"unknown joint {name}"}
                if not self.lower[name] - 1e-9 <= value <= self.upper[name] + 1e-9:
                    return {"status": "rejected", "reason": f"{name} would exceed joint range"}
            self.joints.update({name: float(value) for name, value in targets.items()})
            return {"status": "completed", "reason": ""}

    def set_gripper(self, position: float) -> dict[str, Any]:
        with self.lock:
            self.calls.append(("gripper", {"position": position}))
            if not 0.0 <= position <= 1.0:
                return {"status": "rejected", "reason": "gripper position must be between 0 and 1"}
            self.gripper = float(position)
            return {"status": "completed", "reason": ""}

    def home(self) -> dict[str, Any]:
        with self.lock:
            self.calls.append(("home", {}))
            self.joints.update(FAKE_HOME)
            return {"status": "completed", "reason": "", "target": dict(FAKE_HOME)}


def create_app(robot: FakeRobot) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "state_age_s": 0.0, "executor_subscribers": 1, "session_active": False}

    app.get("/state")(robot.state)

    @app.get("/camera/{name}")
    def camera(name: str) -> dict[str, Any]:
        if name not in CAMERAS:
            raise HTTPException(404, f"unknown camera {name}")
        return robot.camera(name)

    @app.post("/move")
    def move(body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        return robot.move(body["joints"])

    @app.post("/gripper")
    def gripper(body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        return robot.set_gripper(float(body["position"]))

    @app.post("/home")
    def home() -> dict[str, Any]:
        return robot.home()

    @app.post("/stop")
    def stop() -> dict[str, Any]:
        return {"status": "completed", "reason": ""}

    return app


class FakeBridgeServer:
    """Serve a :class:`FakeRobot` on 127.0.0.1 from a daemon thread."""

    def __init__(self, port: int = 0) -> None:
        self.robot = FakeRobot()
        config = uvicorn.Config(
            create_app(self.robot),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("fake bridge did not start")
            time.sleep(0.01)
        bound = self._server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{bound}"

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)


__all__ = ["CAMERAS", "FAKE_HOME", "FakeBridgeServer", "FakeRobot"]

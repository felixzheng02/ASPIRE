"""HTTP client for the lab repo's ``aspire_bridge`` node (or the fake bridge).

The bridge listens on 127.0.0.1 and speaks JSON:

* ``GET /health`` -> ``{"ok", "state_age_s", "executor_subscribers", "session_active"}``
* ``GET /state`` -> ``{"joints": {8 names}, "gripper": 0-1, "gripper_pct", "stamp_s", "age_s"}``
* ``GET /camera/{name}`` -> ``{"jpeg_b64", "stamp_s", "age_s"}``; the wrist camera also
  has ``"depth_png_b64"`` (uint16 mm), ``"intrinsics"`` (3x3) and ``"joints"`` (joint
  state at capture time)
* ``POST /move {"joints": {name: absolute}}``, ``POST /gripper {"position": 0-1}``,
  ``POST /home {}``, ``POST /stop {}`` -> ``{"status", "reason", ...}``. Status is the
  executor's ``completed``, ``stopped_short``, ``timeout`` or ``rejected``.

Joints are URDF conventions in metres and radians; the base is an odom pose.
"""

from __future__ import annotations

from typing import Any

import requests

DEFAULT_URL = "http://127.0.0.1:8230"
#: Reads are local and small; motion calls wait for the executor's result,
#: which itself times out after 10 s per command (Home sends several).
READ_TIMEOUT_S = 5.0
MOTION_TIMEOUT_S = 120.0


class BridgeError(RuntimeError):
    """The bridge was unreachable or answered with an HTTP error."""


class BridgeClient:
    def __init__(self, base_url: str = DEFAULT_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        # Localhost only; never route through an HTTP proxy from the environment.
        self._session.trust_env = False

    def _call(self, method: str, path: str, payload: dict | None, timeout: float) -> dict[str, Any]:
        try:
            response = self._session.request(
                method, f"{self.base_url}{path}", json=payload, timeout=timeout
            )
        except requests.RequestException as exc:
            raise BridgeError(f"bridge {method} {path} failed: {exc}") from exc
        if response.status_code != 200:
            raise BridgeError(
                f"bridge {method} {path} returned {response.status_code}: {response.text}"
            )
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._call("GET", "/health", None, READ_TIMEOUT_S)

    def state(self) -> dict[str, Any]:
        return self._call("GET", "/state", None, READ_TIMEOUT_S)

    def camera(self, name: str) -> dict[str, Any]:
        return self._call("GET", f"/camera/{name}", None, READ_TIMEOUT_S)

    def move(self, joints: dict[str, float]) -> dict[str, Any]:
        return self._call("POST", "/move", {"joints": joints}, MOTION_TIMEOUT_S)

    def gripper(self, position: float) -> dict[str, Any]:
        return self._call("POST", "/gripper", {"position": float(position)}, MOTION_TIMEOUT_S)

    def home(self) -> dict[str, Any]:
        return self._call("POST", "/home", {}, MOTION_TIMEOUT_S)

    def stop(self) -> dict[str, Any]:
        return self._call("POST", "/stop", {}, READ_TIMEOUT_S)

    def close(self) -> None:
        self._session.close()


__all__ = ["DEFAULT_URL", "BridgeClient", "BridgeError"]

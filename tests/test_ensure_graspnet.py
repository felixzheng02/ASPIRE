from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import subprocess
import threading

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ensure_graspnet.sh"


class _Handler(BaseHTTPRequestHandler):
    ok_path = "/healthz"

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == self.ok_path:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


class _OpenApiOnly(_Handler):
    ok_path = "/openapi.json"


def _serve(handler: type[BaseHTTPRequestHandler]) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=str(REPO_ROOT),
        env=merged,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_ensure_graspnet_reuses_healthz() -> None:
    server = _serve(_Handler)
    host, port = server.server_address[:2]
    try:
        result = _run(
            {
                "CAPX_GRASPNET_HOST": str(host),
                "CAPX_GRASPNET_PORT": str(port),
                "CAPX_GRASPNET_WAIT_S": "2",
                "CAPX_GRASPNET_PYTHON": "/nonexistent/python",
            }
        )
    finally:
        server.shutdown()
    assert result.returncode == 0, result.stderr
    assert "already running" in result.stdout


def test_ensure_graspnet_refuses_openapi_only_listener() -> None:
    server = _serve(_OpenApiOnly)
    host, port = server.server_address[:2]
    try:
        result = _run(
            {
                "CAPX_GRASPNET_HOST": str(host),
                "CAPX_GRASPNET_PORT": str(port),
                "CAPX_GRASPNET_WAIT_S": "2",
                "CAPX_GRASPNET_PYTHON": "/nonexistent/python",
            }
        )
    finally:
        server.shutdown()
    assert result.returncode != 0
    assert "GraspNet is not running on :8115" in result.stderr


def test_ensure_graspnet_fails_fast_when_the_child_exits(tmp_path: Path) -> None:
    stub = tmp_path / "crash_python"
    stub.write_text("#!/bin/sh\necho 'import error' >&2\nexit 1\n")
    stub.chmod(0o755)
    log = tmp_path / "graspnet.log"
    result = _run(
        {
            "CAPX_GRASPNET_HOST": "127.0.0.1",
            "CAPX_GRASPNET_PORT": "1",
            "CAPX_GRASPNET_WAIT_S": "30",
            "CAPX_GRASPNET_PYTHON": str(stub),
            "CAPX_GRASPNET_LOG": str(log),
        }
    )
    assert result.returncode != 0
    assert "GraspNet is not running on :8115" in result.stderr
    assert "exited before /healthz" in result.stderr


def test_ensure_graspnet_missing_venv_on_empty_port() -> None:
    result = _run(
        {
            "CAPX_GRASPNET_HOST": "127.0.0.1",
            "CAPX_GRASPNET_PORT": "1",
            "CAPX_GRASPNET_WAIT_S": "1",
            "CAPX_GRASPNET_PYTHON": "/nonexistent/python",
        }
    )
    assert result.returncode != 0
    assert "GraspNet is not running on :8115" in result.stderr
    assert "bootstrap_graspnet.sh" in result.stderr

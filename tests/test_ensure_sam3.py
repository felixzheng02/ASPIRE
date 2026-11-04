from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import subprocess
import threading

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ensure_sam3.sh"


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


class _FakeMatcher(_Handler):
    ok_path = "/health"


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


def test_ensure_sam3_reuses_healthz() -> None:
    server = _serve(_Handler)
    host, port = server.server_address[:2]
    try:
        result = _run(
            {
                "CAPX_SAM3_HOST": str(host),
                "CAPX_SAM3_PORT": str(port),
                "CAPX_SAM3_WAIT_S": "2",
                "CAPX_SAM3_PYTHON": "/nonexistent/python",
            }
        )
    finally:
        server.shutdown()
    assert result.returncode == 0, result.stderr
    assert "already running" in result.stdout


def test_ensure_sam3_refuses_non_sam3_listener() -> None:
    server = _serve(_FakeMatcher)
    host, port = server.server_address[:2]
    try:
        result = _run(
            {
                "CAPX_SAM3_HOST": str(host),
                "CAPX_SAM3_PORT": str(port),
                "CAPX_SAM3_WAIT_S": "2",
                "CAPX_SAM3_PYTHON": "/nonexistent/python",
            }
        )
    finally:
        server.shutdown()
    assert result.returncode != 0
    assert "SAM3 is not running on :8114" in result.stderr


def test_ensure_sam3_fails_fast_when_the_child_exits(tmp_path: Path) -> None:
    stub = tmp_path / "crash_python"
    stub.write_text("#!/bin/sh\necho 'GatedRepoError: Please log in.' >&2\nexit 1\n")
    stub.chmod(0o755)
    log = tmp_path / "sam3.log"
    result = _run(
        {
            "CAPX_SAM3_HOST": "127.0.0.1",
            "CAPX_SAM3_PORT": "1",
            "CAPX_SAM3_WAIT_S": "30",
            "CAPX_SAM3_PYTHON": str(stub),
            "CAPX_SAM3_LOG": str(log),
        }
    )
    assert result.returncode != 0
    assert "SAM3 is not running on :8114" in result.stderr
    assert "exited before /healthz" in result.stderr
    assert "hf auth login" in result.stderr


def test_ensure_sam3_missing_venv_on_empty_port() -> None:
    result = _run(
        {
            "CAPX_SAM3_HOST": "127.0.0.1",
            "CAPX_SAM3_PORT": "1",
            "CAPX_SAM3_WAIT_S": "1",
            "CAPX_SAM3_PYTHON": "/nonexistent/python",
        }
    )
    assert result.returncode != 0
    assert "SAM3 is not running on :8114" in result.stderr
    assert "bootstrap_sam3.sh" in result.stderr


def test_run_experiment_rejects_unknown_perception() -> None:
    script = REPO_ROOT / "run_experiment.sh"
    env = os.environ.copy()
    env["ROBOCASA_PERCEPTION"] = "nope"
    result = subprocess.run(
        ["bash", str(script), "OpenDrawer"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert "must be sam3 or gt" in result.stderr

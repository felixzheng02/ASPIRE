from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from cap_harness.artifacts import RunTiming
from cap_harness.doctor import DoctorConfig, _http_status, check_services
import cap_harness.run as run_module
from cap_harness.run import (
    SAM3_UNAVAILABLE,
    Sam3ServiceUnavailable,
    require_sam3_health,
)
from cap_harness.runtime import ProgramExecutionResult


class _HealthzHandler(BaseHTTPRequestHandler):
    path_ok = "/healthz"

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == self.path_ok:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


class _OpenApiOnlyHandler(_HealthzHandler):
    path_ok = "/openapi.json"


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def test_require_sam3_health_accepts_healthz() -> None:
    server, url = _serve(_HealthzHandler)
    try:
        require_sam3_health(base_url=url, timeout_s=1.0)
    finally:
        server.shutdown()


def test_require_sam3_health_rejects_openapi_only() -> None:
    server, url = _serve(_OpenApiOnlyHandler)
    try:
        with pytest.raises(Sam3ServiceUnavailable, match=SAM3_UNAVAILABLE):
            require_sam3_health(base_url=url, timeout_s=1.0)
    finally:
        server.shutdown()


def test_require_sam3_health_rejects_down_port() -> None:
    with pytest.raises(Sam3ServiceUnavailable, match=SAM3_UNAVAILABLE):
        require_sam3_health(base_url="http://127.0.0.1:1", timeout_s=0.2)


def test_robocasa_run_aborts_before_the_program(tmp_path, monkeypatch) -> None:
    metadata = SimpleNamespace(
        suite_name="OpenDrawer",
        task_id=0,
        task_name="OpenDrawer",
        horizon=10,
    )
    monkeypatch.setattr(
        "cap_harness.robocasa.registry.RobocasaTaskRegistry",
        lambda: SimpleNamespace(resolve=lambda *args: metadata),
    )

    def boom(base_url=None, timeout_s=2.0):
        del base_url, timeout_s
        raise Sam3ServiceUnavailable(SAM3_UNAVAILABLE)

    monkeypatch.setattr(run_module, "require_sam3_health", boom)
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")

    with pytest.raises(Sam3ServiceUnavailable, match=SAM3_UNAVAILABLE):
        run_module.run_program(
            benchmark="robocasa",
            suite="OpenDrawer",
            task_id=0,
            seed=101,
            program_path=program,
            output_root=tmp_path / "outputs",
            max_steps=10,
        )
    assert not list((tmp_path / "outputs").glob("**/*"))


def test_robocasa_run_proceeds_when_healthz_is_up(tmp_path, monkeypatch) -> None:
    metadata = SimpleNamespace(
        suite_name="OpenDrawer",
        task_id=0,
        task_name="OpenDrawer",
        horizon=10,
    )

    class Adapter:
        instances: list = []

        def __init__(self, **kwargs) -> None:
            del kwargs
            self.instances.append(self)

        def reset(self, metadata, seed: int) -> None:
            del metadata, seed

        def check_success(self) -> bool:
            return False

        def close(self) -> None:
            return None

    class Recorder:
        def __init__(self, *, output_root: Path, **kwargs) -> None:
            del kwargs
            self.root = output_root / "run"
            (self.root / "logs").mkdir(parents=True)
            self.timing = RunTiming()
            self.frequency = 20.0
            self.terminal_reason = None
            self.terminated = False
            self.truncated = False

        def save_program(self, path: Path) -> None:
            del path

        def import_model_trace(self, path: Path | None) -> None:
            del path

        def finalize(self, **kwargs) -> None:
            del kwargs

        def close_incomplete(self, error: BaseException, **kwargs) -> None:
            raise AssertionError(f"unexpected incomplete run: {error!r}, {kwargs!r}")

    monkeypatch.setattr(
        "cap_harness.robocasa.registry.RobocasaTaskRegistry",
        lambda: SimpleNamespace(resolve=lambda *args: metadata),
    )
    monkeypatch.setattr("cap_harness.robocasa.adapter.RobocasaAdapter", Adapter)
    monkeypatch.setattr(run_module, "RunRecorder", Recorder)
    monkeypatch.setattr(run_module, "Sam3Provider", lambda **kwargs: object())
    monkeypatch.setattr(run_module, "ContactGraspNetProvider", lambda: object())
    monkeypatch.setattr(run_module, "PyRokiProvider", lambda: object())
    monkeypatch.setattr(run_module, "CuRoboProvider", lambda **kwargs: object())
    monkeypatch.setattr(run_module, "TracedProvider", lambda provider, *args: provider)
    monkeypatch.setattr(
        run_module,
        "CapApi",
        lambda *args, **kwargs: SimpleNamespace(register_tools=lambda registry: registry),
    )
    monkeypatch.setattr(
        run_module,
        "ProgramExecutor",
        lambda tools, recorder=None: SimpleNamespace(
            execute_program=lambda source: ProgramExecutionResult(
                ok=True, result=source, stdout="", calls=()
            )
        ),
    )
    monkeypatch.setattr(run_module, "require_sam3_health", lambda **kwargs: None)
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")

    outcome = run_module.run_program(
        benchmark="robocasa",
        suite="OpenDrawer",
        task_id=0,
        seed=101,
        program_path=program,
        output_root=tmp_path / "outputs",
        max_steps=10,
    )
    assert outcome.program_ok is True


def test_robocasa_run_forwards_endpoint_from_the_environment(tmp_path, monkeypatch) -> None:
    metadata = SimpleNamespace(
        suite_name="OpenDrawer",
        task_id=0,
        task_name="OpenDrawer",
        horizon=10,
    )
    seen: list[dict] = []

    class Adapter:
        def __init__(self, **kwargs) -> None:
            seen.append(kwargs)

        def reset(self, metadata, seed: int) -> None:
            del metadata, seed

        def check_success(self) -> bool:
            return False

        def close(self) -> None:
            return None

    class Recorder:
        def __init__(self, *, output_root: Path, **kwargs) -> None:
            del kwargs
            self.root = output_root / "run"
            (self.root / "logs").mkdir(parents=True)
            self.timing = RunTiming()
            self.frequency = 20.0
            self.terminal_reason = None
            self.terminated = False
            self.truncated = False

        def save_program(self, path: Path) -> None:
            del path

        def import_model_trace(self, path: Path | None) -> None:
            del path

        def finalize(self, **kwargs) -> None:
            del kwargs

        def close_incomplete(self, error: BaseException, **kwargs) -> None:
            raise AssertionError(f"unexpected incomplete run: {error!r}, {kwargs!r}")

    monkeypatch.setattr(
        "cap_harness.robocasa.registry.RobocasaTaskRegistry",
        lambda: SimpleNamespace(resolve=lambda *args: metadata),
    )
    monkeypatch.setattr("cap_harness.robocasa.adapter.RobocasaAdapter", Adapter)
    monkeypatch.setattr(run_module, "RunRecorder", Recorder)
    monkeypatch.setattr(run_module, "Sam3Provider", lambda **kwargs: object())
    monkeypatch.setattr(run_module, "ContactGraspNetProvider", lambda: object())
    monkeypatch.setattr(run_module, "PyRokiProvider", lambda: object())
    monkeypatch.setattr(run_module, "CuRoboProvider", lambda **kwargs: object())
    monkeypatch.setattr(run_module, "TracedProvider", lambda provider, *args: provider)
    monkeypatch.setattr(
        run_module,
        "CapApi",
        lambda *args, **kwargs: SimpleNamespace(register_tools=lambda registry: registry),
    )
    monkeypatch.setattr(
        run_module,
        "ProgramExecutor",
        lambda tools, recorder=None: SimpleNamespace(
            execute_program=lambda source: ProgramExecutionResult(
                ok=True, result=source, stdout="", calls=()
            )
        ),
    )
    monkeypatch.setattr(run_module, "require_sam3_health", lambda **kwargs: None)
    monkeypatch.setenv("CAP_HARNESS_ROBOCASA_ENDPOINT", "ipc:///tmp/capx-sim-7.sock")
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")

    outcome = run_module.run_program(
        benchmark="robocasa",
        suite="OpenDrawer",
        task_id=0,
        seed=101,
        program_path=program,
        output_root=tmp_path / "outputs",
        max_steps=10,
    )
    assert outcome.program_ok is True
    assert seen[0]["endpoint"] == "ipc:///tmp/capx-sim-7.sock"


def test_doctor_probes_sam3_on_healthz(monkeypatch) -> None:
    calls = []

    def http_status(host, port, timeout, path="/openapi.json"):
        calls.append((host, port, timeout, path))
        return True, 200, None

    monkeypatch.setattr("cap_harness.doctor._http_status", http_status)
    check = check_services(DoctorConfig(service_timeout_s=0.25))
    assert check.status == "pass"
    assert ("127.0.0.1", 8114, 0.25, "/healthz") in calls
    assert ("127.0.0.1", 8115, 0.25, "/openapi.json") in calls


def test_http_status_default_path_is_openapi() -> None:
    # Signature used by the existing mock in test_doctor.py.
    assert _http_status.__defaults__[-1] == "/openapi.json"

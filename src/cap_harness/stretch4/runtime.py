"""System 2 MCP runtime for ``--station stretch4``, mirroring ``build_yam_runtime``.

Reuses YAM's ``RuntimeControl`` (one actuation worker) and ``LiveYamAdapter``
(camera snapshots for ``observe``); neither is YAM-specific. Not supported on
this station: cuRobo (no collision-aware planner for this arm), recording,
and the Viser page (its 3D model is the YAM station).
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any

from cap_harness.agent.embedded import EmbeddedAgentRuntime

from .adapter import Stretch4Adapter
from .kinematics import Stretch4IKProvider
from .station import WRIST_CAMERA, Stretch4Station, build_station


def build_stretch4_runtime(
    station: Stretch4Station,
    *,
    allow_motion: bool = False,
    **api_options: Any,
) -> tuple[Any, EmbeddedAgentRuntime]:
    """Attach to a bridge-backed station; construction performs no motion."""
    from cap_harness.api import CapApi
    from cap_harness.contracts import TaskContext
    from cap_harness.registry import STRETCH4_PUBLIC_TOOL_NAMES, ToolRegistry
    from cap_harness.yam_real.control.runtime import RuntimeControl
    from cap_harness.yam_real.live_adapter import LiveYamAdapter

    raw = Stretch4Adapter(
        station,
        allow_physical_motion=allow_motion,
        task_context=TaskContext(
            suite="system2",
            task_id=0,
            task_name="system2",
            language="Follow the operator goal.",
            family="stretch4",
        ),
    )
    control = RuntimeControl(raw)
    adapter = LiveYamAdapter(raw, control)
    try:
        api = CapApi(
            adapter,
            default_camera=WRIST_CAMERA,
            ik_providers={"mink": Stretch4IKProvider(station.kinematics)},
            **api_options,
        )
        registry = api.register_tools(
            ToolRegistry(public_extension_allowlist=STRETCH4_PUBLIC_TOOL_NAMES)
        )
        runtime = EmbeddedAgentRuntime(api, registry, operation_controller=control)
    except BaseException:
        control.close()
        raise
    return adapter, runtime


def serve(args: Any) -> int:
    """``python -m cap_harness.agent --station stretch4 [--sim] ...``."""
    import uvicorn

    from cap_harness.agent.server import create_app

    unsupported = [
        flag
        for flag, value in (("--record-dir", args.record_dir), ("--curobo-url", args.curobo_url))
        if value
    ]
    if unsupported:
        raise SystemExit(f"--station stretch4 does not support {', '.join(unsupported)}")
    with ExitStack() as stack:
        bridge_url = args.bridge_url
        if args.sim:
            from .fake_bridge import FakeBridgeServer

            fake = FakeBridgeServer()
            stack.callback(fake.close)
            bridge_url = fake.url
        station = build_station(bridge_url)
        options: dict[str, Any] = {}
        if args.sam3_url:
            from cap_harness.providers.sam3 import Sam3Provider

            options["segmentation_provider"] = Sam3Provider(base_url=args.sam3_url)
        adapter, runtime = build_stretch4_runtime(
            station, allow_motion=args.allow_motion, **options
        )
        stack.callback(adapter.close)
        stack.callback(runtime.close)
        uvicorn.run(create_app(runtime.session), host=args.host, port=args.port)
    return 0


__all__ = ["build_stretch4_runtime", "serve"]

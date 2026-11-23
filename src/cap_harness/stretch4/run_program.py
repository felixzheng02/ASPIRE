"""Run one CaP program file on the Stretch 4 through the System 2 session.

    .venv-system2/bin/python -m cap_harness.stretch4.run_program examples/stretch4/observe.py
    .venv-system2/bin/python -m cap_harness.stretch4.run_program --allow-motion examples/stretch4/go_home.py

Uses the same session, program executor and motion interlock as the MCP
runtime, so do not run it while that runtime is connected to the same bridge.
``--sim`` uses the in-process fake bridge instead. The exit status says whether the
program ran; each motion's own ``ok`` and error are in the printed result.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from .client import DEFAULT_URL
from .runtime import build_stretch4_runtime
from .station import build_station


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("program", type=Path)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--bridge-url", default=DEFAULT_URL)
    parser.add_argument("--sim", action="store_true", help="Use the in-process fake bridge")
    args = parser.parse_args(argv)
    with ExitStack() as stack:
        url = args.bridge_url
        if args.sim:
            from .fake_bridge import FakeBridgeServer

            fake = FakeBridgeServer()
            stack.callback(fake.close)
            url = fake.url
        adapter, runtime = build_stretch4_runtime(
            build_station(url), allow_motion=args.allow_motion
        )
        stack.callback(adapter.close)
        stack.callback(runtime.close)
        session = runtime.session
        job_id = session.start_program(args.program.read_text(), label=args.program.name)
        result = session.wait(job_id, timeout_s=None, detail="full")
        print(json.dumps(result, indent=2, default=str))
        program = (result or {}).get("result") or {}
        return 0 if result and not result.get("error") and program.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

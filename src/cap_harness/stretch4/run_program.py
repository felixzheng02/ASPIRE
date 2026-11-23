"""Run one CaP program file on the Stretch 4 through the System 2 session.

    .venv-system2/bin/python -m cap_harness.stretch4.run_program examples/stretch4/observe.py
    .venv-system2/bin/python -m cap_harness.stretch4.run_program --allow-motion examples/stretch4/go_home.py

Uses the same program executor, motion interlock and control worker as the
MCP runtime, so do not run it while that runtime is connected to the same
bridge. ``--sim`` uses the in-process fake bridge instead. The exit status says
whether the program ran; each motion's own ``ok`` and error are in the printed
result.

Programs run in the CaP sandbox, which has no ``argv`` and no file access, so:

* ``--set NAME=VALUE`` defines the string variable ``NAME`` before the program runs;
* ``--sam3-url URL`` connects SAM3 for ``localize_object``;
* ``--save-images DIR`` writes each array in the program's ``result["images"]``
  (name -> HxWx3 uint8 image or HxW boolean mask) to ``DIR/<name>.png``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import ExitStack
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from .client import DEFAULT_URL
from .runtime import build_stretch4_runtime
from .station import build_station


def _definition(text: str) -> str:
    name, separator, value = text.partition("=")
    if not separator or not name.isidentifier() or name.startswith("_"):
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return f"{name} = {value!r}\n"


def _jsonable(value: Any) -> Any:
    """Readable JSON: small arrays as lists, large ones as their shape."""
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return np.round(value.astype(float), 6).tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _save_images(images: Any, directory: Path) -> list[str]:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, image in dict(images or {}).items():
        array = np.asarray(image)
        if array.dtype == bool:
            array = array.astype(np.uint8) * 255
        path = directory / f"{name}.png"
        Image.fromarray(array.astype(np.uint8)).save(path)
        written.append(str(path))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("program", type=Path)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--bridge-url", default=DEFAULT_URL)
    parser.add_argument("--sim", action="store_true", help="Use the in-process fake bridge")
    parser.add_argument(
        "--set",
        dest="definitions",
        type=_definition,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Define a string variable for the program",
    )
    parser.add_argument("--save-images", type=Path, metavar="DIR", help='Save result["images"]')
    parser.add_argument("--sam3-url", help="SAM3 service, needed by localize_object")
    args = parser.parse_args(argv)
    with ExitStack() as stack:
        url = args.bridge_url
        if args.sim:
            from .fake_bridge import FakeBridgeServer

            fake = FakeBridgeServer()
            stack.callback(fake.close)
            url = fake.url
        options = {}
        if args.sam3_url:
            from cap_harness.providers.sam3 import Sam3Provider

            options["segmentation_provider"] = Sam3Provider(base_url=args.sam3_url)
        adapter, runtime = build_stretch4_runtime(
            build_station(url), allow_motion=args.allow_motion, **options
        )
        stack.callback(adapter.close)
        stack.callback(runtime.close)
        source = "".join(args.definitions) + args.program.read_text()
        # Called directly rather than as a session job so the raw result (and
        # its images) comes back. Motion still goes through the live adapter's
        # interlock and single control worker.
        execution = runtime.session.executor.execute_program(source, filename=str(args.program))
        result = execution.result
        images = None
        if isinstance(result, Mapping):  # the sandbox returns a read-only mapping
            result = dict(result)
            images = result.pop("images", None)
        saved = _save_images(images, args.save_images) if args.save_images and images else []
        print(execution.stdout, end="")  # the program's own prints, readable
        print(
            json.dumps(
                _jsonable(
                    {
                        "ok": execution.ok,
                        "error": execution.error,
                        "result": result,
                        "saved_images": saved,
                    }
                ),
                indent=2,
            )
        )
        return 0 if execution.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

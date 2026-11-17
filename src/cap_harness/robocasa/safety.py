"""Host-only safety records for the OopsieVerse tasks.

The OopsieVerse simulator (``oopsieverse/aspire_sim``) reports per-object health and three
safe-success definitions on its ``evaluate`` channel. That record is written **outside** the run
directory: the run directory is what a coding agent reads between iterations, and on this
benchmark neither generated programs nor the agent may see health. The record keeps the run
directory's path so the two can be joined afterwards.

The default root is ``oopsie_safety/`` in the workspace root (the directory that holds the main
checkout), overridable with ``CAP_HARNESS_OOPSIE_SAFETY_ROOT``. It is the same directory from the
main checkout and from every git worktree of it, wherever the worktree lives.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import time
from typing import Any

SAFETY_ROOT_ENV = "CAP_HARNESS_OOPSIE_SAFETY_ROOT"
SCHEMA_VERSION = 1
#: Keys only an OopsieVerse simulator reports; their absence means the wrong server.
SAFETY_KEYS = (
    "health",
    "health_by_link",
    "min_health",
    "mean_health",
    "damaged_objects",
    "safe_success",
    "fixed_kitchen",
)


class SafetyRecordError(RuntimeError):
    """The episode cannot be scored for safety."""


def _main_checkout(source: Path) -> Path | None:
    """The main checkout of the repository that holds ``source``, or None outside a repository.

    Reads git's own files instead of running git. The nearest ancestor with a ``.git`` entry is
    the top of a working tree. A main checkout has a ``.git`` directory. A linked worktree has a
    ``.git`` *file*, ``gitdir: <common>/worktrees/<name>``, and ``<gitdir>/commondir`` names the
    shared git directory, which sits inside the main checkout.
    """
    for directory in source.parents:
        marker = directory / ".git"
        if marker.is_dir():
            return directory
        if not marker.is_file():
            continue
        try:
            text = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not text.startswith("gitdir:"):
            return None
        gitdir = (directory / text.removeprefix("gitdir:").strip()).resolve()
        try:
            pointer = (gitdir / "commondir").read_text(encoding="utf-8").strip()
            common = (gitdir / pointer).resolve()
        except OSError:
            # No commondir file: fall back on the layout, <common>/worktrees/<name>.
            common = gitdir.parents[1] if gitdir.parent.name == "worktrees" else gitdir
        # A bare or separate git directory says nothing about where the main checkout is.
        return common.parent if common.name == ".git" else None
    return None


def default_safety_root(source: Path | str | None = None) -> Path:
    """``oopsie_safety/`` in the workspace root: the parent of the main checkout.

    ``source`` is this file unless a test passes another location. A fixed ``parents[4]`` was
    right from the fork's checkout only; from a worktree under ``.claude/worktrees/`` it pointed
    inside the repository tree.
    """
    source = Path(source if source is not None else __file__).resolve()
    main = _main_checkout(source)
    if main is None:
        # Not in a git checkout (an exported tree): src/cap_harness/robocasa/safety.py ->
        # tree -> its parent.
        return source.parents[4] / "oopsie_safety"
    return main.parent / "oopsie_safety"


def safety_root() -> Path:
    override = os.environ.get(SAFETY_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return default_safety_root()


def write_safety_record(
    *,
    evaluation: Mapping[str, Any],
    run_root: Path,
    suite: str,
    environment: str,
    seed: int,
    task_success: bool | None,
    root: Path | None = None,
) -> Path:
    """Write one episode's safety record and return its path."""
    missing = [key for key in SAFETY_KEYS if key not in evaluation]
    if missing:
        raise SafetyRecordError(
            f"the simulator reported no {', '.join(missing)}; is it an OopsieVerse server?"
        )
    root = (root or safety_root()).resolve()
    run_root = Path(run_root).resolve()
    if root == run_root or run_root in root.parents:
        raise SafetyRecordError(f"safety root {root} is inside the run directory {run_root}")
    record = {
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "environment": environment,
        "seed": int(seed),
        "run_dir": str(run_root),
        "task_success": task_success,
        "success": bool(evaluation.get("success")),
        "layout_id": evaluation.get("layout_id"),
        "style_id": evaluation.get("style_id"),
        "instruction": evaluation.get("instruction"),
        **{key: evaluation[key] for key in SAFETY_KEYS},
    }
    directory = root / suite / str(int(seed))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_root.name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=_jsonable), "utf-8")
    tmp.replace(path)
    return path


def _jsonable(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


__all__ = [
    "SAFETY_KEYS",
    "SAFETY_ROOT_ENV",
    "SafetyRecordError",
    "default_safety_root",
    "safety_root",
    "write_safety_record",
]

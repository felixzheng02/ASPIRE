"""Typed discovery for the supported RoboCasa CaP tasks.

RoboCasa is reached over the CaP-X ZeroMQ wire protocol rather than imported, so this module
holds no MuJoCo handles and the environment names below are the strings the *simulator process*
builds (``server.py --task <environment>``), not classes resolved here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RobocasaRegistryError(ValueError):
    """Raised when a RoboCasa task reference is invalid."""


@dataclass(frozen=True, slots=True)
class RobocasaTaskMetadata:
    task_name: str
    environment: str
    language: str
    robots: tuple[str, ...]
    camera_names: tuple[str, ...]
    env_configuration: str | None = None
    task_id: int = 0
    family: str = "robocasa"
    horizon: int | None = None
    """Episode length the task's own authors specify, in control ticks.

    RoboCasa's kitchen environments state ``horizon=1000`` (``kitchen.py``), which is what
    belongs here. The simulator process is constructed with a larger ``max_steps`` so that one
    server can serve harnesses with different budgets; a task scored against a horizon its
    authors did not choose fails for a reason that has nothing to do with the policy.
    """

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    @property
    def arms(self) -> tuple[str, ...]:
        return ("primary",) if len(self.robots) == 1 else ("primary", "secondary")

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "camera_names": list(self.camera_names),
            "environment": self.environment,
            "family": self.family,
            "language": self.language,
            "robots": list(self.robots),
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


#: ``agentview`` is ASPIRE's portable name for the scene camera, as every shipped example and
#: the localize skill spell it; the simulator maps it to RoboCasa's ``robot0_agentview_left``.
#: ``robot0_eye_in_hand`` needs no mapping -- RoboCasa and ASPIRE already agree on it.
_MOBILE_CAMERAS = ("agentview", "robot0_eye_in_hand")

#: RoboCasa's own default, in 20 Hz control ticks.
_ROBOCASA_HORIZON = 1000

#: Lowest development seed, from ASPIRE's Robosuite partition: development 101-125,
#: unseen-check 126-130, held-out 1-100 (`.claude/skills/iterative-debugging/SKILL.md`).
DEV_SEED_FLOOR = 101


def scene_group_for_seed(seed: int) -> str:
    """Which RoboCasa scene group a seed draws its kitchen from.

    The partition is ASPIRE's; the groups are RoboCasa's. Development and unseen-check seeds
    (>= 101) draw from ``TRAIN`` (layouts and styles 11-60) and held-out seeds (1-100) from
    ``TEST`` (1-10). The two groups are disjoint, so a program tuned on development scenes is
    evaluated on kitchens it has never been shown -- which is what makes the held-out number a
    claim about generalization rather than about seed arithmetic.

    This is deliberately a stronger split than ASPIRE's tabletop benchmarks use, where
    development and held-out seeds differ only in value and come from one distribution. The
    consequence, accepted knowingly: ASPIRE's own Robosuite success rates are not a reference
    point for RoboCasa numbers produced this way.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RobocasaRegistryError("seed must be a non-negative integer")
    return "train" if seed >= DEV_SEED_FLOOR else "test"

ROBOCASA_TASKS: tuple[RobocasaTaskMetadata, ...] = (
    RobocasaTaskMetadata(
        "OpenDrawer",
        "OpenDrawer",
        "Open the drawer.",
        ("PandaOmron",),
        _MOBILE_CAMERAS,
        horizon=_ROBOCASA_HORIZON,
    ),
    RobocasaTaskMetadata(
        "PickPlaceCounterToStove",
        "PickPlaceCounterToStove",
        "Pick the object from the counter and place it on the stove.",
        ("PandaOmron",),
        _MOBILE_CAMERAS,
        horizon=_ROBOCASA_HORIZON,
    ),
)
"""The task set verified end-to-end on this simulator.

``task_name`` deliberately equals ``environment`` rather than carrying a separate snake_case
suite name as Robosuite does: here the same string is the CLI's ``--suite``, the registry key and
the simulator's ``--task``, and one spelling across the stack is worth more than the convention.
"""


class RobocasaTaskRegistry:
    """Resolve only the audited RoboCasa task set."""

    def __init__(self, tasks: tuple[RobocasaTaskMetadata, ...] = ROBOCASA_TASKS) -> None:
        self._tasks = {task.task_name: task for task in tasks}
        if len(self._tasks) != len(tasks):
            raise RobocasaRegistryError("RoboCasa task names must be unique")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def enumerate_tasks(self) -> tuple[RobocasaTaskMetadata, ...]:
        return tuple(self._tasks.values())

    def resolve(
        self, task_ref: str | tuple[str, int] | RobocasaTaskMetadata, task_id: int | None = None
    ) -> RobocasaTaskMetadata:
        if isinstance(task_ref, RobocasaTaskMetadata):
            return task_ref
        if isinstance(task_ref, tuple):
            if len(task_ref) != 2:
                raise RobocasaRegistryError("task tuple must contain (task_name, task_id)")
            task_name, tuple_id = task_ref
            if task_id is not None and task_id != tuple_id:
                raise RobocasaRegistryError("conflicting RoboCasa task ids")
            task_id = int(tuple_id)
        else:
            task_name = task_ref
            if ":" in task_name:
                task_name, encoded_id = task_name.rsplit(":", 1)
                if task_id is not None and task_id != int(encoded_id):
                    raise RobocasaRegistryError("conflicting RoboCasa task ids")
                task_id = int(encoded_id)
        if task_id is None:
            task_id = 0
        if task_id != 0:
            raise RobocasaRegistryError("each RoboCasa task currently has task_id 0")
        try:
            return self._tasks[str(task_name)]
        except KeyError as exc:
            raise RobocasaRegistryError(
                f"unknown RoboCasa task {task_name!r}; available: {', '.join(self.available_tasks)}"
            ) from exc


__all__ = [
    "DEV_SEED_FLOOR",
    "ROBOCASA_TASKS",
    "RobocasaRegistryError",
    "RobocasaTaskMetadata",
    "RobocasaTaskRegistry",
    "scene_group_for_seed",
]

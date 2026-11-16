"""Grasp-only benchmark: how often does the grasp API actually pick the object up?

One episode per seed. Segment the object named in the instruction, ask
``generate_grasps`` for candidates, try the top three by score in order, and after each
attempt ask whether the object came off the surface. Nothing else: no placing, no base
motion, no task success. The only question is whether
``generate_grasps`` -> ``solve_ik`` -> ``move_to_pose`` -> ``close_gripper`` -> lift works.

"Picked" is measured through the public API alone -- the object is re-segmented after the
lift and its cloud centroid must have risen. The simulator's own ``evaluate`` channel is
never consulted; it scores placing, which is not what this measures.

Run from ``aspire/`` with the simulator serving the task and SAM3, GraspNet and PyRoki up::

    PYTHONPATH="" .venv-robocasa/bin/python examples/robocasa/grasp_bench/grasp_bench.py \
        --seeds 101-150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from cap_harness.api import CapApi
from cap_harness.contracts import Pose
from cap_harness.providers.curobo.client import CuRoboProvider
from cap_harness.providers.graspnet.client import ContactGraspNetProvider
from cap_harness.providers.pyroki.client import PyRokiProvider
from cap_harness.providers.sam3.client import Sam3Provider
from cap_harness.registry import ROBOCASA_PUBLIC_TOOL_NAMES, ToolRegistry
from cap_harness.robocasa.adapter import RobocasaAdapter
from cap_harness.robocasa.registry import RobocasaTaskRegistry

CAMERA = "agentview"
BASE_FRAME = "robot0_link0"
STANDOFF_M = 0.10
LIFT_M = 0.12
PICKED_RISE_M = 0.05
TOLERANCE_M = 0.02
MAX_STEPS = 150


def build_tools() -> tuple[RobocasaAdapter, dict]:
    """The same public tool namespace a generated program is given."""
    adapter = RobocasaAdapter(registry=RobocasaTaskRegistry(), horizon=1000)
    curobo = CuRoboProvider()
    api = CapApi(
        adapter,
        segmentation_provider=Sam3Provider(),
        grasp_providers={"contact-graspnet": ContactGraspNetProvider()},
        ik_providers={"pyroki": PyRokiProvider(), "curobo": curobo},
        trajectory_planning_providers={"curobo": curobo},
        integrated_pose_planning_providers={"curobo-integrated": curobo},
        default_camera=CAMERA,
    )
    registry = api.register_tools(
        ToolRegistry(public_extension_allowlist=ROBOCASA_PUBLIC_TOOL_NAMES)
    )
    return adapter, dict(registry.public_tools())


def target_noun(instruction: str) -> str:
    """The object the task names, e.g. "Pick the fish from the plate ..." -> "fish"."""
    match = re.search(
        r"pick (?:up )?the (.+?) (?:from|on|off|and|into|in) ", instruction, re.IGNORECASE
    )
    return match.group(1).strip() if match else "object"


def object_centroid(tools: dict, noun: str) -> tuple[np.ndarray | None, int]:
    """Centroid of the best mask for ``noun``, in robot0_link0, via depth back-projection."""
    found = tools["segment_text"](CAMERA, noun)
    if not found.ok or not found.segmentations:
        return None, 0
    segmentation = found.segmentations[0]
    try:
        cloud = tools["mask_to_point_cloud"](segmentation, CAMERA, BASE_FRAME)
    except ValueError:
        return None, 0
    return np.asarray(cloud.points).mean(axis=0), len(cloud.points)


def offset_pose(pose: Pose, delta: np.ndarray) -> Pose:
    return Pose(
        position=np.asarray(pose.position, dtype=np.float64) + delta,
        quaternion_wxyz=np.asarray(pose.quaternion_wxyz, dtype=np.float64),
        frame=pose.frame,
    )


def is_reachable(tools: dict, candidate) -> bool:
    """Both the standoff and the grasp pose must solve, or the attempt cannot happen."""
    approach = np.asarray(candidate.pose.as_matrix())[:3, 2]
    standoff = offset_pose(candidate.pose, -approach * STANDOFF_M)
    return bool(tools["solve_ik"](standoff).ok) and bool(tools["solve_ik"](candidate.pose).ok)


def feasible_candidates(tools: dict, candidates: list, want: int) -> tuple[list, int]:
    """Walk candidates best-first and keep the first ``want`` that IK can actually reach.

    Pure computation -- no simulator ticks are spent. Feasibility is judged from the pose the
    arm is in now, so it can go stale after a failed attempt moves the arm; ``attempt_grasp``
    re-checks and will still record ``unreachable`` if it does.
    """
    kept = []
    for tested, candidate in enumerate(candidates, start=1):
        if is_reachable(tools, candidate):
            kept.append(candidate)
            if len(kept) == want:
                return kept, tested
    return kept, len(candidates)


def attempt_grasp(tools: dict, candidate, noun: str) -> dict:
    """One pick attempt. Returns what happened, including whether the object came up."""
    before, _ = object_centroid(tools, noun)
    approach = np.asarray(candidate.pose.as_matrix())[:3, 2]
    standoff = offset_pose(candidate.pose, -approach * STANDOFF_M)

    reachable = bool(tools["solve_ik"](standoff).ok) and bool(tools["solve_ik"](candidate.pose).ok)
    if not reachable:
        return {"outcome": "unreachable", "picked": False, "score": float(candidate.score)}

    tools["open_gripper"]()
    approach_move = tools["move_to_pose"](standoff, TOLERANCE_M, MAX_STEPS)
    grasp_move = tools["move_to_pose"](candidate.pose, TOLERANCE_M, MAX_STEPS)
    tools["close_gripper"]()
    jaw = float(tools["get_robot_state"]().gripper_positions["primary"])

    end_effector = tools["get_robot_state"]().end_effector_poses["primary"]
    lift = tools["move_to_pose"](
        offset_pose(end_effector, np.array([0.0, 0.0, LIFT_M])), 0.03, MAX_STEPS
    )
    after, _ = object_centroid(tools, noun)

    rise = None if before is None or after is None else float(after[2] - before[2])
    picked = rise is not None and rise > PICKED_RISE_M
    return {
        "outcome": "picked" if picked else "not_picked",
        "picked": picked,
        "score": float(candidate.score),
        "approach_ok": bool(approach_move.ok),
        "grasp_ok": bool(grasp_move.ok),
        "lift_ok": bool(lift.ok),
        "jaw_after_close": round(jaw, 4),
        "object_rise_m": None if rise is None else round(rise, 4),
    }


def segment_and_rank(tools: dict, noun: str) -> tuple[object | None, list]:
    """Segment the object and rank its grasp candidates best-first. Nothing else."""
    found = tools["segment_text"](CAMERA, noun)
    if not found.ok or not found.segmentations:
        return None, []
    segmentation = found.segmentations[0]
    grasps = tools["generate_grasps"](CAMERA, segmentation, max_candidates=20)
    return segmentation, (sorted(grasps.grasps, key=lambda c: -c.score) if grasps.ok else [])


def ask_where_to_stand(tools: dict, ranked: list) -> dict:
    """``robocasa.plan_base_offset`` over the live candidates. Spends no simulator ticks.

    ``standoff_m`` matches this file's own ``is_reachable``, so "reachable" means the same
    thing to the query as it does to ``--ik-filter``'s feasible count.
    """
    offer = tools["robocasa.plan_base_offset"](
        [candidate.pose for candidate in ranked],
        standoff_m=STANDOFF_M,
        min_reachable=1,
    )
    return {
        "ok": bool(offer["ok"]),
        "dy": round(float(offer["dy"]), 4),
        "reachable_now": int(offer["reachable_now"]),
        "reachable_after": int(offer["reachable_after"]),
        "offsets_tested": int(offer["offsets_tested"]),
        "offsets_blocked": sum(1 for entry in offer["offsets"] if entry["free"] is False),
    }


def run_episode(
    adapter: RobocasaAdapter,
    tools: dict,
    task: str,
    seed: int,
    attempts: int,
    ik_filter: bool = False,
    base_reach: str = "off",
) -> dict:
    started = time.perf_counter()
    adapter.reset(task, seed)
    instruction = str(tools["get_task_context"]().language)
    noun = target_noun(instruction)
    record: dict = {"seed": seed, "instruction": instruction, "object": noun}

    segmentation, ranked = segment_and_rank(tools, noun)
    if segmentation is None:
        record.update(failure="no_mask", picked=False, attempts=[])
        record["seconds"] = round(time.perf_counter() - started, 1)
        return record
    record["mask_score"] = round(float(segmentation.score), 4)
    record["n_candidates"] = len(ranked)

    # Where should the robot stand? Answered before anything moves, and recorded even in
    # "plan" mode, so one sweep yields both the stand-still baseline and the after-number.
    if base_reach != "off" and ranked:
        record["base_offset"] = ask_where_to_stand(tools, ranked)
        offer = record["base_offset"]
        if base_reach == "move" and offer["ok"] and offer["dy"] != 0.0:
            moved = tools["robocasa.navigate_to_pose"](0.0, offer["dy"], 0.0)
            # agentview is robot-mounted, so the move invalidates the mask and the
            # candidates with it. Re-perceiving is the real cost of repositioning.
            after_segmentation, after_ranked = segment_and_rank(tools, noun)
            record["base_move"] = {
                "ok": bool(moved.ok),
                "steps": int(moved.steps_executed),
                "mask_score_before": record["mask_score"],
                "mask_score_after": (
                    None
                    if after_segmentation is None
                    else round(float(after_segmentation.score), 4)
                ),
                "n_candidates_after": len(after_ranked),
            }
            if after_segmentation is not None and after_ranked:
                segmentation, ranked = after_segmentation, after_ranked
                record["n_candidates"] = len(ranked)

    if ik_filter and ranked:
        candidates, tested = feasible_candidates(tools, ranked, attempts)
        record["n_ik_tested"] = tested
        record["n_ik_feasible"] = len(candidates)
    else:
        candidates = ranked[:attempts]
    if not candidates:
        record.update(failure="no_grasps", picked=False, attempts=[])
        record["seconds"] = round(time.perf_counter() - started, 1)
        return record

    tried = []
    for index, candidate in enumerate(candidates):
        outcome = attempt_grasp(tools, candidate, noun)
        outcome["rank"] = index
        tried.append(outcome)
        if outcome["picked"]:
            break
    record.update(
        attempts=tried,
        picked=any(a["picked"] for a in tried),
        failure=None if any(a["picked"] for a in tried) else "no_attempt_succeeded",
    )
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        first, last = text.split("-", 1)
        return list(range(int(first), int(last) + 1))
    return [int(part) for part in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="PickPlaceCounterToStove")
    parser.add_argument("--seeds", default="101-150")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--ik-filter",
        action="store_true",
        help="rank candidates best-first but only attempt those solve_ik can reach, "
        "so the attempt budget is spent on feasible poses instead of unreachable ones",
    )
    parser.add_argument(
        "--base-reach",
        choices=("off", "plan", "move"),
        default="off",
        help="off: unchanged. plan: also record robocasa.plan_base_offset's answer per "
        "episode, which costs no simulator ticks and reports both the stand-still count and "
        "the count a base move would reach. move: additionally drive the base to the "
        "suggested offset and re-perceive, which measures what repositioning costs",
    )
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    arguments = parser.parse_args(argv)

    seeds = parse_seeds(arguments.seeds)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"

    adapter, tools = build_tools()
    records: list[dict] = []
    try:
        with results_path.open("w", encoding="utf-8") as stream:
            for position, seed in enumerate(seeds, start=1):
                try:
                    record = run_episode(
                        adapter,
                        tools,
                        arguments.task,
                        seed,
                        arguments.attempts,
                        ik_filter=arguments.ik_filter,
                        base_reach=arguments.base_reach,
                    )
                except Exception as exc:  # one bad episode must not end the sweep
                    record = {
                        "seed": seed,
                        "picked": False,
                        "failure": f"raised: {type(exc).__name__}: {exc}",
                        "attempts": [],
                    }
                records.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                picked = sum(1 for r in records if r.get("picked"))
                print(
                    f"[{position}/{len(seeds)}] seed {seed} "
                    f"object={record.get('object', '?')!r} "
                    f"candidates={record.get('n_candidates', 0)} "
                    f"-> {'PICKED' if record.get('picked') else record.get('failure')} "
                    f"| running {picked}/{len(records)} "
                    f"({record.get('seconds', 0)}s)",
                    flush=True,
                )
    finally:
        adapter.close()

    picked = [r for r in records if r.get("picked")]
    reasons: dict[str, int] = {}
    for record in records:
        if not record.get("picked"):
            reasons[str(record.get("failure"))] = reasons.get(str(record.get("failure")), 0) + 1
    by_rank: dict[int, int] = {}
    for record in picked:
        rank = next(a for a in record["attempts"] if a["picked"])["rank"]
        by_rank[rank] = by_rank.get(rank, 0) + 1

    # The headline number for base reachability: of the scenes that produced candidates at
    # all, how many have at least one the arm can actually reach -- standing still, versus
    # standing where plan_base_offset says to. Both come from IK, so neither costs a tick.
    with_candidates = [r for r in records if r.get("n_candidates", 0)]
    offers = [r["base_offset"] for r in with_candidates if r.get("base_offset")]
    base_reach_summary = (
        {
            "scenes_with_candidates": len(with_candidates),
            "reachable_standing_still": sum(1 for o in offers if o["reachable_now"] >= 1),
            "reachable_after_offset": sum(1 for o in offers if o["reachable_after"] >= 1),
            "scenes_offered_a_move": sum(1 for o in offers if o["dy"] != 0.0),
            "scenes_with_no_free_offset": sum(
                1 for o in offers if not o["ok"] and o["reachable_now"] == 0
            ),
        }
        if offers
        else None
    )
    moves = [r["base_move"] for r in records if r.get("base_move")]
    base_move_summary = (
        {
            "moves_attempted": len(moves),
            "moves_converged": sum(1 for m in moves if m["ok"]),
            "mean_steps": round(float(np.mean([m["steps"] for m in moves])), 1),
            "lost_the_mask_after_moving": sum(
                1 for m in moves if m["mask_score_after"] is None or m["n_candidates_after"] == 0
            ),
            "mean_mask_score_before": round(
                float(np.mean([m["mask_score_before"] for m in moves])), 4
            ),
            "mean_mask_score_after": round(
                float(
                    np.mean(
                        [m["mask_score_after"] for m in moves if m["mask_score_after"] is not None]
                    )
                ),
                4,
            )
            if any(m["mask_score_after"] is not None for m in moves)
            else None,
        }
        if moves
        else None
    )

    summary = {
        "task": arguments.task,
        "ik_filter": bool(arguments.ik_filter),
        "base_reach": arguments.base_reach,
        "base_reachability": base_reach_summary,
        "base_motion_cost": base_move_summary,
        "seeds": f"{seeds[0]}-{seeds[-1]}",
        "episodes": len(records),
        "picked": len(picked),
        "success_rate": round(len(picked) / len(records), 4) if records else 0.0,
        "succeeded_on_rank": by_rank,
        "failure_reasons": reasons,
        "episodes_with_no_candidates": sum(1 for r in records if r.get("n_candidates", 0) == 0),
        "episodes_with_no_feasible_candidate": sum(
            1 for r in records if r.get("n_candidates", 0) and not r.get("n_ik_feasible", 1)
        ),
        "mean_candidates": round(float(np.mean([r.get("n_candidates", 0) for r in records])), 2)
        if records
        else 0.0,
        "total_minutes": round(sum(r.get("seconds", 0) for r in records) / 60.0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

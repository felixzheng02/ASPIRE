#!/usr/bin/env python3
"""Stage RoboCasa skill proposals for the verifier.

Writes ``$RUN_ROOT/verification/<benchmark>__<suite>__task_<id>/proposed_skills.md``
from the task's ``skill_report.json``. Does not inject tabletop addenda and does
not dispatch a verifier. Do not use parent ``dispatch.py`` on a RoboCasa campaign.

``proposed_new`` fields copied into the markdown: ``skill``, ``title``,
``trigger``, ``code``, ``evidence``. ``proposed_edits`` fields: ``skill``,
``section``, ``change``, ``why``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from campaign import (
    heldout_seeds,
    resolve_run_root,
    verification_dir,
)
from campaign import task_dir as campaign_task_dir


def stage_proposals(
    run_root: Path, benchmark: str, suite: str, task_id: int, out: Path
) -> int:
    """Write the verifier's copy of the proposals, and return how many there are."""
    report_path = campaign_task_dir(run_root, benchmark, suite, task_id) / "skill_report.json"
    if not report_path.is_file():
        raise SystemExit(
            f"no skill_report.json for {suite}/task_{task_id} — nothing was proposed, so there "
            "is nothing to verify. Skip this task rather than dispatching a verifier."
        )
    report = json.loads(report_path.read_text())
    new = report.get("proposed_new") or []
    edits = report.get("proposed_edits") or []
    if not new and not edits:
        raise SystemExit(f"{suite}/task_{task_id} proposed nothing — no verifier needed.")

    manifests = sorted(
        (campaign_task_dir(run_root, benchmark, suite, task_id) / "validation" / "runs").glob(
            "*/manifest.json"
        )
    )
    heldout = heldout_seeds(run_root)
    if manifests:
        payload = json.loads(manifests[-1].read_text())
        passes = payload.get("passes", "?")
        heldout_n = len(heldout)
    else:
        passes, heldout_n = "?", len(heldout)

    lines = [
        "# Skills Under Test\n",
        f"Proposals from `{suite}/task_{task_id}`. **They are staged here and are deliberately "
        "NOT in the shared library** — whether they get promoted is what your verdict decides. "
        "Do not go looking for them under `skills/`; this file is their only home. Re-solve the "
        "task from the library PLUS this file, without seeing the author's program.\n",
        f"The frozen program scores **{passes}/{heldout_n}** held-out. Read that before drawing "
        "conclusions from your own score: on a task whose residual failure is kinematic, a low "
        "score of your own is expected and is **not** by itself a refutation. Judge each proposal "
        "on whether it transmits and whether its mechanism reproduces.\n",
        "---\n",
    ]
    for index, proposal in enumerate(new, 1):
        lines.append(
            f"## {index}. → `{proposal['skill']}` — {proposal['title']}\n\n"
            f"**Trigger.** {proposal['trigger']}\n"
        )
        lines.append("```python\n" + (proposal.get("code") or "").strip() + "\n```\n")
        lines.append(f"**Evidence.** {proposal.get('evidence', '')}\n\n---\n")
    if edits:
        lines.append("## Edits\n")
        for edit in edits:
            lines.append(f"### `{edit['skill']}` § {edit.get('section', '')}\n")
            lines.append(f"**Change.** {edit.get('change', '')}\n\n*Why:* {edit.get('why', '')}\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return len(new) + len(edits)


def main() -> None:
    """CLI: stage ``proposed_skills.md`` for one RoboCasa task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="robocasa")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args()
    run_root = resolve_run_root(args.run_root)
    verify_dir = verification_dir(run_root, args.benchmark, args.suite, args.task_id)
    count = stage_proposals(
        run_root, args.benchmark, args.suite, args.task_id, verify_dir / "proposed_skills.md"
    )
    print(f"staged {count} proposal(s) -> {verify_dir / 'proposed_skills.md'}")


if __name__ == "__main__":
    main()

# RoboCasa Skill Library

This directory is the **only** skill library for a RoboCasa campaign. Workers consult it,
grade it in `skill_report.json`, and propose new entries. They do **not** edit it. A
verifier re-solves the task from the library plus those proposals, without the proposer's
program. The coordinator promotes **only** `verified` entries into these files.

Do **not** edit the tabletop parent `.claude/skills/iterative-debugging/skills/` from a
RoboCasa campaign. Do **not** copy tabletop snippets here: they assume `robot_base` /
LIBERO and will send the arm ~14 m away or at the wrong hand frame.

| Skill | Purpose |
|---|---|
| [localize.md](localize.md) | SAM3 prompts and 3D localization in `robot0_link0` |
| [grasp.md](grasp.md) | Grasp selection on observed RGB-D in `robot0_link0` |
| [navigation.md](navigation.md) | Wheeled base: relative `navigate_to_pose`, world `base_pose_is_free` |
| [manipulation.md](manipulation.md) | Contact with articulated kitchen parts via public APIs |

All snippets use only the documented public CaP program APIs (`docs/api-reference.md` and
`docs/robocasa.md`): no imports, no NumPy, no simulator internals, and a top-level `result`
variable per program. Programs may `import math` and nothing else.

Promoted findings should include provenance: source suite/task, development seeds, held-out
result when available, source code path, and date. Avoid universal claims from one kitchen.

## How an entry gets here

Nothing is written into these files by the agent that discovered it. An entry arrives only by
this route, and every step is on disk:

```
subagent proposes  →  verifier reproduces the fix   →  coordinator promotes
(skill_report.json)   from the proposal ALONE          (edits this directory)
                      (verification_report.json)
```

The verifier gets this library and the proposal, and is denied the proposer's program,
findings, and run artifacts. If it cannot reach the fix on development seed 101 within two
attempts, the proposal is **refuted** and stays out. So an entry here means: *written down, on
its own, this was enough for an agent that had never seen the solution.*

What that does **not** mean is that the entry generalizes. Proposals are verified on the task
they came from, so passing shows the text transmits, not that it transfers. Evidence of
transfer accumulates separately, as later tasks grade the entry `useful` in their own reports
— which is why the evidence lines matter and why they should name every task, not just the
first.

Entries are also graded on the way out. Every subagent reports each section it opened as
`useful`, `misleading`, `wrong`, or `not-applicable`, and the coordinator repairs what comes
back bad before dispatching more work. An entry that keeps drawing `misleading` usually has a
trigger problem — it is being read by tasks it was never meant for — and the fix is to narrow
the trigger, not to bolt a warning onto the end.

There are no verified RoboCasa entries yet. Stubs exist so `proposed_new.skill` can name an
existing `*.md` file (`validate_skill_report` requires that).

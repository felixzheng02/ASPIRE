"""The RoboCasa fix-loop skill folder: tabletop product, embodiment constraints."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude/skills/iterative-debugging/robocasa"
PARENT = ROOT / ".claude/skills/iterative-debugging/SKILL.md"
CODEX = ROOT / ".codex/skills/iterative-debugging/SKILL.md"


def _read(name: str) -> str:
    return (SKILL / name).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _without_benign(text: str) -> str:
    # "512" and "8115" contain the substring "51"; they are not LIBERO seeds.
    return text.replace("512", "").replace("8115", "")


def test_robocasa_skill_folder_is_complete_and_has_no_library() -> None:
    for name in ("SKILL.md", "subagent-prompt.md", "main-agent-prompt.md", "clean-task-slate.md"):
        assert (SKILL / name).is_file(), name
    assert not (SKILL / "skills").exists()
    assert not (SKILL / "score_campaign.py").exists()
    assert not (SKILL / "verifier-prompt.md").exists()


def test_robocasa_skill_is_the_tabletop_product_with_embodiment_constraints() -> None:
    overview = _read("SKILL.md")
    subagent = _read("subagent-prompt.md")
    coordinator = _read("main-agent-prompt.md")
    slate = _read("clean-task-slate.md")
    texts = (overview, subagent, coordinator, slate)

    assert "fix_code.py" in overview and "fix_code.py" in subagent and "fix_code.py" in coordinator
    assert "run_validation.py" in overview and "run_validation.py" in coordinator
    assert "# Code block" not in overview and "# Code block" not in subagent
    assert "skill-library-frozen" not in overview

    for text in (overview, subagent, coordinator):
        assert ".venv-robocasa" in text or "ensure_sam3.sh" in text
    assert ".venv-robocasa/bin/cap-harness" in subagent
    assert "robocasa.navigate_to_pose" in subagent
    assert "--init-mode saved" in subagent or "INIT_MODE=saved" in subagent
    assert "--workers 1" in subagent and "--recycle-after 0" in subagent
    assert "--idle-timeout 0" in subagent
    assert "ensure_sam3.sh" in overview and "ensure_sam3.sh" in subagent
    assert "/healthz" in overview and "/healthz" in subagent and "/healthz" in coordinator
    assert "Never kill SAM3" in subagent or "never kill SAM3" in subagent.lower()
    assert "Never kill SAM3" in slate or "never kill SAM3" in slate.lower()

    for text in texts:
        assert "--workers 4" not in text
        assert "--perception-port" not in text
        assert "ROBOCASA_PERCEPTION" not in text
        assert "fake matcher" not in text.lower()
        assert "ground truth SAM" not in text
        assert "solve_ik" not in text
        assert "51" not in _without_benign(text)

    for text in (overview, subagent, coordinator):
        assert "101" in text and "102" in text and "103" in text
        assert "126" in text
    assert "1 2 3" in overview or "1–3" in overview or "1-3" in overview

    parent = PARENT.read_text(encoding="utf-8")
    assert "robocasa/SKILL.md" in parent
    assert "robocasa/SKILL.md" in CODEX.read_text(encoding="utf-8")


def test_robocasa_worker_owns_the_simulator_not_sam3() -> None:
    subagent = _read("subagent-prompt.md")
    coordinator = _read("main-agent-prompt.md")
    slate = _read("clean-task-slate.md")
    assert "simulator.pid" in subagent and "simulator.pid" in coordinator
    assert "server.py" in subagent and "--task OpenDrawer" in subagent
    assert "pkill" in subagent.lower() or "never `pkill`" in subagent
    assert "pkill" in slate
    assert "dispatch.py" in coordinator and "Do not use parent" in _flat(coordinator)
    assert "Non-privileged API audit: PASS/QUARANTINED" in subagent
    assert "never rewrite it to hide its provenance" in subagent
    assert "Non-privileged API audit" in coordinator

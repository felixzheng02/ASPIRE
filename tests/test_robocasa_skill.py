"""The RoboCasa fix-loop skill folder: tabletop product, embodiment constraints."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude/skills/iterative-debugging/robocasa"
PARENT = ROOT / ".claude/skills/iterative-debugging/SKILL.md"
CODEX = ROOT / ".codex/skills/iterative-debugging/SKILL.md"
SCRIPTS = ROOT / ".claude/skills/iterative-debugging/scripts"
ROBOCASA_SKILLS = ".claude/skills/iterative-debugging/robocasa/skills"

_SKILL_FILES = (
    "SKILL.md",
    "subagent-prompt.md",
    "main-agent-prompt.md",
    "clean-task-slate.md",
    "verifier-prompt.md",
    "evosearch.md",
    "skills/README.md",
    "skills/localize.md",
    "skills/grasp.md",
    "skills/navigation.md",
    "skills/manipulation.md",
)


def _read(name: str) -> str:
    return (SKILL / name).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _without_benign(text: str) -> str:
    # "512" and "8115" contain the substring "51"; they are not LIBERO seeds.
    return text.replace("512", "").replace("8115", "")


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robocasa_skill_folder_is_complete() -> None:
    for name in _SKILL_FILES:
        assert (SKILL / name).is_file(), name
    assert (SKILL / "skills/README.md").is_file()
    assert (SKILL / "verifier-prompt.md").is_file()
    assert not (SKILL / "score_campaign.py").exists()
    assert "skill-library-frozen" not in _read("SKILL.md")


def test_robocasa_skill_is_the_tabletop_product_with_embodiment_constraints() -> None:
    overview = _read("SKILL.md")
    subagent = _read("subagent-prompt.md")
    coordinator = _read("main-agent-prompt.md")
    slate = _read("clean-task-slate.md")
    texts = tuple(_read(name) for name in _SKILL_FILES)

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
    assert "ensure_graspnet.sh" in slate

    for text in texts:
        assert "--workers 4" not in text
        assert "--perception-port" not in text
        assert "ROBOCASA_PERCEPTION" not in text
        assert "fake matcher" not in text.lower()
        assert "ground truth SAM" not in text
        assert "51" not in _without_benign(text)
    assert "solve_ik" in subagent and "move_to_pose" in subagent
    assert "generate_grasps" in subagent and "select_grasp" in subagent
    assert "plan_motion" in subagent
    assert "base_pose_is_free" in subagent and "plan_standoff_pose" in subagent
    assert "plan_approach_pose" in subagent
    assert "ensure_pyroki.sh" in overview and "ensure_pyroki.sh" in subagent
    assert "ensure_graspnet.sh" in overview and "ensure_graspnet.sh" in subagent
    assert "ensure_curobo.sh" in overview and "ensure_curobo.sh" in subagent

    for text in (overview, subagent, coordinator):
        assert "101" in text and "102" in text and "103" in text
        assert "126" in text
        assert "get_task_context().language" in text
    assert "1 2 3" in overview or "1–3" in overview or "1-3" in overview

    parent = PARENT.read_text(encoding="utf-8")
    assert "robocasa/SKILL.md" in parent
    assert "--evosearch" in parent
    assert "robocasa/SKILL.md" in CODEX.read_text(encoding="utf-8")


def test_robocasa_library_verifier_and_evosearch_flag() -> None:
    overview = _read("SKILL.md")
    subagent = _read("subagent-prompt.md")
    coordinator = _read("main-agent-prompt.md")
    slate = _read("clean-task-slate.md")
    verifier = _read("verifier-prompt.md")
    evosearch = _read("evosearch.md")

    assert "skill_report.json" in overview and "skill_report.json" in subagent
    assert "skill_report.json" in coordinator
    assert "verifier-prompt.md" in overview and "verifier-prompt.md" in coordinator
    assert "--evosearch" in overview and "--evosearch" in coordinator
    assert "default off" in overview.lower() or "default **off**" in overview.lower()
    assert "has no `skills/` library" not in overview
    assert "No verifier, no skill promotion" not in coordinator
    assert "There is no skill library and no `skill_report.json`" not in subagent
    assert "no shared skill library to edit" not in slate

    assert "--skills-dir" in coordinator
    assert ROBOCASA_SKILLS in coordinator
    assert ROBOCASA_SKILLS in overview
    assert "--skills-dir" in subagent
    assert "--skill-library-sha" in subagent
    assert "--validate-skill-report" in subagent

    assert "--parallel-per-gpu 1" in evosearch or "--parallel-per-gpu` 1" in evosearch
    assert "--workers 1" in evosearch
    assert "campaign.json" in evosearch
    assert "only when" in evosearch.lower()

    assert "101" in verifier
    assert "robocasa/skills" in verifier or ROBOCASA_SKILLS in verifier
    assert "--workers 1" in verifier
    assert "simulator.pid" in verifier
    assert "never `pkill`" in verifier.lower() or "never pkill" in verifier.lower()
    assert "does **not** mean the skill generalizes" in verifier
    assert '--suite "$SUITE"' in verifier
    assert "get_task_context().language" in verifier
    assert "--suite OpenDrawer" not in verifier
    assert '--suite "$SUITE"' in evosearch
    assert "--suite OpenDrawer" not in evosearch

    assert "does not delete `skills/`" in slate or "does not delete skills/" in slate.lower()


def test_robocasa_worker_owns_the_simulator_not_sam3() -> None:
    subagent = _read("subagent-prompt.md")
    coordinator = _read("main-agent-prompt.md")
    slate = _read("clean-task-slate.md")
    assert "simulator.pid" in subagent and "simulator.pid" in coordinator
    assert "server.py" in subagent and '--task "$SUITE"' in subagent
    assert '--suite "$SUITE"' in subagent
    assert "get_task_context().language" in subagent
    assert "--task OpenDrawer" not in subagent
    assert "--suite OpenDrawer" not in subagent
    assert "pkill" in subagent.lower() or "never `pkill`" in subagent
    assert "pkill" in slate
    assert "dispatch.py" in coordinator and "Do not use parent" in _flat(coordinator)
    assert "Non-privileged API audit: PASS/QUARANTINED" in subagent
    assert "never rewrite it to hide its provenance" in subagent
    assert "Non-privileged API audit" in coordinator


def test_init_run_help_lists_evosearch_and_skills_dir(capsys, monkeypatch) -> None:
    init_run = _load_script("init_run")
    monkeypatch.setattr(sys, "argv", ["init_run.py", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        init_run.main()
    assert exc_info.value.code in (0, None)
    out = capsys.readouterr().out
    assert "--evosearch" in out
    assert "--skills-dir" in out


def test_init_run_records_evosearch_and_skills_dir(tmp_path: Path, monkeypatch) -> None:
    init_run = _load_script("init_run")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps([{"benchmark": "robocasa", "suite": "OpenDrawer", "task_id": 0}])
    )
    runs = tmp_path / "runs"
    monkeypatch.setattr(init_run, "RUNS_ROOT", runs)
    monkeypatch.setattr(init_run, "LATEST_LINK", runs / "LATEST")
    monkeypatch.delenv("CAP_HARNESS_ROBOCASA_EVOSEARCH", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "init_run.py",
            "--tasks",
            str(tasks),
            "--dev-seeds",
            "101",
            "102",
            "103",
            "--unseen-seeds",
            "126",
            "--heldout-count",
            "3",
            "--skills-dir",
            ROBOCASA_SKILLS,
            "--evosearch",
        ],
    )
    init_run.main()
    settings = json.loads((runs / "LATEST").resolve().joinpath("campaign.json").read_text())
    assert settings["evosearch"] is True
    assert settings["skills_dir"] == ROBOCASA_SKILLS
    assert settings["dev_seeds"] == [101, 102, 103]


def test_init_run_defaults_evosearch_off(tmp_path: Path, monkeypatch) -> None:
    init_run = _load_script("init_run")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps([{"benchmark": "robocasa", "suite": "OpenDrawer", "task_id": 0}])
    )
    runs = tmp_path / "runs"
    monkeypatch.setattr(init_run, "RUNS_ROOT", runs)
    monkeypatch.setattr(init_run, "LATEST_LINK", runs / "LATEST")
    monkeypatch.delenv("CAP_HARNESS_ROBOCASA_EVOSEARCH", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "init_run.py",
            "--tasks",
            str(tasks),
            "--dev-seeds",
            "101",
            "--unseen-seeds",
            "126",
            "--heldout-count",
            "3",
        ],
    )
    init_run.main()
    settings = json.loads((runs / "LATEST").resolve().joinpath("campaign.json").read_text())
    assert settings["evosearch"] is False
    assert "skills_dir" not in settings

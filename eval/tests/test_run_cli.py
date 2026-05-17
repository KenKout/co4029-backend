from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from eval.run import main


def test_help_lists_all_args(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--scenarios",
        "--backend",
        "--budget-usd",
        "--judge-model",
        "--output",
        "--dry-run",
    ):
        assert flag in out


def test_missing_budget_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--scenarios", "all", "--backend", "new"])
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "budget-usd" in err or "required" in err


def test_negative_budget_refused(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    rc = main(["--budget-usd", "-1", "--output", str(output)])
    assert rc == 2
    assert "non-negative" in capsys.readouterr().err


def test_zero_budget_runs_with_no_scenarios(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "result.json"
    rc = main(["--budget-usd", "0", "--output", str(output)])
    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["budget_usd"] == 0.0
    assert payload["spent_usd"] == 0.0
    assert payload["results"] == []


def test_dry_run_no_llm_calls(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "dry.json"
    rc = main(["--budget-usd", "5", "--dry-run", "--output", str(output)])
    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["dry_run"] is True
    assert payload["spent_usd"] == 0.0


def test_default_backend_is_new(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "result.json"
    rc = main(["--budget-usd", "1", "--output", str(output)])
    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["backend"] == "new"


def test_invalid_backend_choice_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--budget-usd", "1", "--backend", "sideways"])
    assert exc.value.code != 0


def test_scenarios_filter_parses_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "result.json"
    rc = main(
        [
            "--budget-usd",
            "1",
            "--scenarios",
            "quiz_generation,interview_generation",
            "--output",
            str(output),
        ]
    )
    assert rc == 0


def test_module_invocation_help_via_subprocess() -> None:
    import subprocess

    backend_new = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "eval.run", "--help"],
        cwd=backend_new,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0
    for flag in (
        "--scenarios",
        "--backend",
        "--budget-usd",
        "--judge-model",
        "--output",
        "--dry-run",
    ):
        assert flag in result.stdout

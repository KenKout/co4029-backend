from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from eval.runners import REGISTRY
from eval.spec import ScenarioSpec

EVAL_DIR = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = EVAL_DIR / "scenarios"
FIXTURES_DIR = EVAL_DIR / "fixtures"

EXPECTED_FRONTMATTER_KEYS = {
    "fixture_id",
    "material_type",
    "expected_chunks",
    "language",
    "license",
}

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


FIXTURE_NON_FIXTURE_FILES = {"LICENSE.md", "README.md"}


def _all_scenario_paths() -> list[Path]:
    return sorted(SCENARIOS_DIR.glob("*.yaml"))


def _all_fixture_paths() -> list[Path]:
    return sorted(p for p in FIXTURES_DIR.glob("*.md") if p.name not in FIXTURE_NON_FIXTURE_FILES)


def _parse_frontmatter(text: str) -> dict[str, Any]:
    m = FRONTMATTER_RE.match(text)
    if m is None:
        return {}
    parsed = yaml.safe_load(m.group(1))
    return parsed if isinstance(parsed, dict) else {}


def _load_spec(path: Path) -> ScenarioSpec:
    raw = yaml.safe_load(path.read_text())
    return ScenarioSpec.model_validate(raw)


@pytest.mark.parametrize("scenario_path", _all_scenario_paths(), ids=lambda p: p.stem)
def test_all_scenarios_parse(scenario_path: Path) -> None:
    spec = _load_spec(scenario_path)
    assert spec.scenario_id == scenario_path.stem, (
        f"scenario_id {spec.scenario_id!r} should match filename {scenario_path.stem!r}"
    )


def test_at_least_four_scenarios_present() -> None:
    paths = _all_scenario_paths()
    assert len(paths) >= 4, f"expected >= 4 scenarios, got {len(paths)}"


@pytest.mark.parametrize("scenario_path", _all_scenario_paths(), ids=lambda p: p.stem)
def test_all_scenarios_reference_valid_fixtures(scenario_path: Path) -> None:
    spec = _load_spec(scenario_path)
    for fixture_id in spec.fixtures:
        matches = list(FIXTURES_DIR.glob(f"{fixture_id}.*"))
        assert matches, (
            f"scenario {spec.scenario_id} references fixture {fixture_id!r} "
            f"but no file matches {FIXTURES_DIR / (fixture_id + '.*')}"
        )


@pytest.mark.parametrize("scenario_path", _all_scenario_paths(), ids=lambda p: p.stem)
def test_capability_in_registry(scenario_path: Path) -> None:
    spec = _load_spec(scenario_path)
    assert spec.capability in REGISTRY, (
        f"scenario {spec.scenario_id} declares capability {spec.capability!r} "
        f"which has no runner registered. Known capabilities: {sorted(REGISTRY)}"
    )


@pytest.mark.parametrize("scenario_path", _all_scenario_paths(), ids=lambda p: p.stem)
def test_threshold_in_valid_range(scenario_path: Path) -> None:
    spec = _load_spec(scenario_path)
    for criterion in spec.criteria:
        assert 1.0 <= criterion.expected_score_threshold <= 5.0, (
            f"scenario {spec.scenario_id} criterion {criterion.id!r} threshold "
            f"{criterion.expected_score_threshold} outside [1.0, 5.0]"
        )


@pytest.mark.parametrize("scenario_path", _all_scenario_paths(), ids=lambda p: p.stem)
def test_scenarios_have_estimated_cost(scenario_path: Path) -> None:
    spec = _load_spec(scenario_path)
    assert spec.estimated_cost_usd_per_fixture > 0.0, (
        f"scenario {spec.scenario_id} must declare a positive "
        f"estimated_cost_usd_per_fixture for the budget pre-check"
    )
    assert spec.total_estimated_cost_usd() > 0.0


@pytest.mark.parametrize("fixture_path", _all_fixture_paths(), ids=lambda p: p.stem)
def test_fixture_frontmatter_valid(fixture_path: Path) -> None:
    text = fixture_path.read_text()
    fm = _parse_frontmatter(text)
    assert fm, f"fixture {fixture_path.name} missing YAML frontmatter"
    missing = EXPECTED_FRONTMATTER_KEYS - set(fm.keys())
    assert not missing, f"fixture {fixture_path.name} frontmatter missing keys: {sorted(missing)}"
    assert fm["fixture_id"] == fixture_path.stem, (
        f"fixture_id {fm['fixture_id']!r} should match filename {fixture_path.stem!r}"
    )


def test_at_least_three_fixtures_present() -> None:
    paths = _all_fixture_paths()
    assert len(paths) >= 3, f"expected >= 3 fixtures, got {len(paths)}"


def test_fixtures_license_file_exists() -> None:
    license_path = FIXTURES_DIR / "LICENSE.md"
    assert license_path.exists(), "fixtures/LICENSE.md required for provenance"
    text = license_path.read_text()
    assert "CC0" in text, "LICENSE.md should declare CC0 / public-domain release"


def test_scenario_spec_rejects_out_of_range_threshold() -> None:
    bad = {
        "scenario_id": "bad",
        "description": "x",
        "capability": "quiz_generation",
        "fixtures": ["beginner_python_loops"],
        "criteria": [
            {
                "id": "c1",
                "description": "x",
                "expected_score_threshold": 6.0,
            }
        ],
        "estimated_cost_usd_per_fixture": 0.1,
    }
    with pytest.raises(ValidationError):
        ScenarioSpec.model_validate(bad)


def test_scenario_spec_rejects_duplicate_criterion_ids() -> None:
    bad = {
        "scenario_id": "bad",
        "description": "x",
        "capability": "quiz_generation",
        "fixtures": ["beginner_python_loops"],
        "criteria": [
            {"id": "c1", "description": "x", "expected_score_threshold": 3.5},
            {"id": "c1", "description": "y", "expected_score_threshold": 3.5},
        ],
        "estimated_cost_usd_per_fixture": 0.1,
    }
    with pytest.raises(ValidationError):
        ScenarioSpec.model_validate(bad)


def test_scenario_spec_rejects_unknown_field() -> None:
    bad = {
        "scenario_id": "bad",
        "description": "x",
        "capability": "quiz_generation",
        "fixtures": ["beginner_python_loops"],
        "criteria": [{"id": "c1", "description": "x", "expected_score_threshold": 3.5}],
        "estimated_cost_usd_per_fixture": 0.1,
        "rogue_field": "nope",
    }
    with pytest.raises(ValidationError):
        ScenarioSpec.model_validate(bad)

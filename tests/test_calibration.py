"""Offline tests for the proxy-vs-judge calibration.

The judges need Ollama, but their scores are frozen in
``data/eval/judge-scores.json``. These tests recompute every published statistic
from that file (no model, no network) and fail if the frozen data, the proxy or
the math drift from docs/eval-calibration.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import calibrate_judge as cj  # noqa: E402


def test_parse_score_maps_the_rubric_to_unit_interval() -> None:
    assert cj.parse_score('{"score": 4}') == 1.0
    assert cj.parse_score('{"score": 3}') == 0.75
    assert cj.parse_score('{"score": 0}') == 0.0


@pytest.mark.parametrize("raw", ['{"score": 5}', '{"score": "4"}', '{"x": 1}', "four", "[]"])
def test_parse_score_rejects_malformed_replies(raw: str) -> None:
    assert cj.parse_score(raw) is None


def test_collect_items_excludes_abstentions_and_unanswerable_cases() -> None:
    items, excluded = cj.collect_items()
    assert items and excluded
    assert not {i.item_id for i in items} & set(excluded)
    assert all(not i.case_id.startswith("ho-ooc") for i in items)


def test_ollama_judge_returns_none_on_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    judge = cj.ollama_judge("m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert judge("answer", "context") is None


def test_ollama_judge_sends_temperature_zero_and_a_schema() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"score": 2}'}})

    judge = cj.ollama_judge("m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert judge("answer", "context") == 0.5
    assert seen["options"] == {"temperature": 0.0, "seed": 0}
    assert "format" in seen


def test_freeze_run_does_not_mutate_its_input() -> None:
    items, excluded = cj.collect_items()
    before: dict[str, Any] = {"items": {}, "runs": {}}
    after = cj.freeze_run(before, items[:2], excluded, "x", {"model": "m"}, {})
    assert before == {"items": {}, "runs": {}}
    assert "x" in after["runs"]
    assert after["runs"]["x"]["prompt_sha256"] == cj.prompt_sha256()


def test_best_cutoff_breaks_ties_toward_the_current_floor() -> None:
    # Cutoffs 0.6 and 0.9 both separate these perfectly; 0.6 is nearer 0.70.
    cutoff, kappa = cj.best_cutoff([0.2, 0.6, 0.9], [0.0, 1.0, 1.0])
    assert kappa == 1.0
    assert cutoff == 0.6


def test_best_cutoff_uses_kappa_not_raw_agreement() -> None:
    """With 3 of 4 faithful, 'everything is faithful' wins raw agreement; kappa rejects it."""
    proxy = [0.1, 0.2, 0.3, 0.9]
    judge = [1.0, 1.0, 0.0, 1.0]
    # Raw agreement: cutoff 0.1 agrees on 3/4 by calling everything faithful
    # (kappa 0); cutoff 0.9 agrees on only 2/4 but carries information (kappa 0.2).
    cutoff, kappa = cj.best_cutoff(proxy, judge)
    assert cutoff == 0.9
    assert kappa == pytest.approx(0.2)


# --- the frozen measurement --------------------------------------------------


@pytest.fixture(scope="module")
def frozen() -> dict[str, Any]:
    return cj.load_frozen()


def test_frozen_scores_reproduce_the_published_numbers(frozen: dict[str, Any]) -> None:
    items, _ = cj.collect_items()
    failures = cj.check(frozen, items)
    assert not failures, "\n".join(failures)


def test_every_published_number_is_pinned(frozen: dict[str, Any]) -> None:
    """A run added to the JSON without updating _EXPECTED would go unchecked."""
    assert set(cj.summary_numbers(frozen)) == set(cj._EXPECTED)


def test_frozen_runs_carry_provenance(frozen: dict[str, Any]) -> None:
    runs = frozen["runs"]
    assert len(runs) >= 2
    for run in runs.values():
        assert run["digest"] != "unknown"
        assert run["options"] == {"temperature": 0.0, "seed": 0}
        assert run["prompt_sha256"] == cj.prompt_sha256()

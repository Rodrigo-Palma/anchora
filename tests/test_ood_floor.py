"""Offline tests for the out-of-domain floor calibration.

The nomic cosines need Ollama, but they are frozen in
``data/eval/ood-similarities.json``. These tests recompute the published
"abstains by floor" tables from that file (no model, no network) and fail if
the frozen data, the lexical overlap or ADR 6 drift apart.
"""

from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import calibrate_ood_floor as cof  # noqa: E402

_PINNED = {
    "ho-defensoria-garantias",
    "ho-defensoria-natureza",
    "ho-lgpd-objetivo",
    "ho-licitacao-substituiu",
}


def _frozen() -> dict[str, Any]:
    return cof.load_frozen()


def test_collect_questions_covers_the_four_labelled_sets() -> None:
    questions = cof.collect_questions()
    sizes = {name: sum(1 for q in questions if q.subset == name) for name in cof.SUBSETS}
    assert sizes == {
        "golden": 24,
        "holdout_answerable": 22,
        "holdout_out_of_corpus": 6,
        "ood": 12,
    }
    assert len({q.case_id for q in questions}) == len(questions)


def test_overlap_table_reproduces_the_2026_10_09_addendum() -> None:
    """Floors 1/2/3 abstain counts match the table ADR 6 already published."""
    rows = cof.overlap_rows(_frozen()["records"])
    assert rows == [
        (1, (0, 1, 4, 11)),
        (2, (0, 4, 6, 12)),
        (3, (3, 9, 6, 12)),
    ]


def test_dense_floor_as_wired_can_only_add_abstentions() -> None:
    """Under the AND wiring, every threshold abstains at least as often as t=0."""
    records = _frozen()["records"]
    rows = cof.cosine_rows(records)
    base = dict(cof.overlap_rows(records))[cof.CURRENT_MIN_OVERLAP]
    assert rows[0][1] == base
    for (_, prev), (_, cur) in pairwise(rows):
        assert all(c >= p for p, c in zip(prev, cur, strict=True))


def test_frozen_file_records_model_identity() -> None:
    model = _frozen()["model"]
    assert model["name"].startswith("nomic-embed-text")
    assert len(model["digest"]) == 64
    assert model["ollama_version"]


def test_pinned_questions_are_reported() -> None:
    pinned = {r["case_id"] for r in _frozen()["records"] if r["pinned"]}
    assert pinned == _PINNED


def test_check_passes_on_the_frozen_file_and_adr() -> None:
    assert cof.check() == []


def test_check_fails_when_an_overlap_drifts(tmp_path: Path) -> None:
    data = _frozen()
    data["records"][0]["distinct_overlap"] += 5
    path = tmp_path / "ood.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    failures = cof.check(frozen_path=path)
    assert any("overlap drift" in f for f in failures)


def test_check_fails_when_the_published_table_is_missing(tmp_path: Path) -> None:
    adr = tmp_path / "adr.md"
    adr.write_text("# nothing published\n", encoding="utf-8")
    failures = cof.check(adr_path=adr)
    assert any("not published" in f for f in failures)


def test_measure_uses_the_injected_similarity(store: Any) -> None:
    questions = [q for q in cof.collect_questions() if q.case_id in {"lai-prazo", "ood-008"}]
    records = cof.measure(questions, store, similarity=lambda _q: 0.5)
    by_id = {r["case_id"]: r for r in records}
    assert by_id["lai-prazo"]["top_cosine"] == 0.5
    assert by_id["lai-prazo"]["distinct_overlap"] >= 2
    assert by_id["ood-008"]["distinct_overlap"] == 1


def test_model_identity_reads_version_digest_and_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.99.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "nomic-embed-text:latest", "digest": "a" * 64}]}
            )
        return httpx.Response(
            200, json={"details": {"family": "nomic-bert", "parameter_size": "137M"}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    model = cof.model_identity("nomic-embed-text", client=client)
    assert model["digest"] == "a" * 64
    assert model["ollama_version"] == "0.99.0"
    assert model["details"]["family"] == "nomic-bert"


def test_model_identity_refuses_a_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [], "version": "1"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    with pytest.raises(RuntimeError, match="not pulled"):
        cof.model_identity("nomic-embed-text", client=client)


# --- Overlap-1 dense rule (pre-registered in ADR 6 before measuring) ---------


def test_t1_follows_the_preregistered_formula() -> None:
    """Highest must-abstain overlap-1 cosine (0.548712) + 0.01, rounded up."""
    assert cof.preregistered_t1(_frozen()["records"]) == 0.56


def test_shipped_threshold_is_the_preregistered_t1() -> None:
    from anchora.config import Settings

    assert Settings().ood_overlap1_similarity_threshold == cof.preregistered_t1(
        _frozen()["records"]
    )


def test_rule_rows_only_move_holdout_answerable() -> None:
    rows = cof.rule_rows(_frozen()["records"])
    assert rows == [("as wired (overlap ≥ 2)", (0, 4, 6, 12)), ("overlap-1 rule", (0, 3, 6, 12))]


def test_rule_meets_the_preregistered_criterion() -> None:
    result = cof.evaluate_rule(_frozen()["records"])
    assert result.failures == []
    assert result.recovered == ("ho-defensoria-natureza",)
    assert result.margin >= cof.REQUIRED_MARGIN


def _with_cosine(case_id: str, value: float) -> list[dict[str, Any]]:
    records = _frozen()["records"]
    return [{**r, "top_cosine": value} if r["case_id"] == case_id else r for r in records]


def test_criterion_rejects_a_margin_below_two_hundredths() -> None:
    # Must-abstain max 0.56 -> t1 0.57; natureza (0.577) recovered by 0.017 only.
    result = cof.evaluate_rule(_with_cosine("ho-ooc-imposto-renda", 0.56))
    assert result.recovered == ("ho-defensoria-natureza",)
    assert any("margin" in f for f in result.failures)


def test_criterion_rejects_when_nothing_is_recovered() -> None:
    result = cof.evaluate_rule(_with_cosine("ood-008", 0.58))
    assert result.recovered == ()
    assert any("recovered" in f for f in result.failures)


def test_criterion_rejects_a_golden_abstention() -> None:
    records = [
        {**r, "distinct_overlap": 1, "top_cosine": 0.1} if r["case_id"] == "lai-prazo" else r
        for r in _frozen()["records"]
    ]
    result = cof.evaluate_rule(records)
    assert any("golden" in f for f in result.failures)


def test_check_fails_when_the_shipped_threshold_drifts(monkeypatch: pytest.MonkeyPatch) -> None:
    from anchora.config import settings

    monkeypatch.setattr(settings, "ood_overlap1_similarity_threshold", 0.5)
    assert any("threshold" in f for f in cof.check())

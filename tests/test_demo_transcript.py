"""Offline checks on the frozen SSE transcript of the local-model demo.

``data/demo/stream-transcript.json`` is one real run of ``POST /ask/stream``
against ``qwen3:32b`` + ``nomic-embed-text`` (recorded by
``scripts/record_stream_demo.py``). The model is not deterministic, so the
transcript is the evidence: these tests check, with no model, that it has the
order and schema ADR 8 promises (``token``+, ``retracted``?, ``done`` with
sources) and that the retraction search is reported as it happened.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from anchora.guardrails import is_abstention, validate_output

_TRANSCRIPT = Path(__file__).resolve().parents[1] / "data" / "demo" / "stream-transcript.json"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DONE_KEYS = {
    "question",
    "sources",
    "grounded",
    "refused",
    "truncated",
    "pii_redacted",
    "output_pii_redacted",
    "trace_id",
    "timing_ms",
}


@pytest.fixture(scope="module")
def transcript() -> dict[str, Any]:
    return json.loads(_TRANSCRIPT.read_text(encoding="utf-8"))


def _runs(transcript: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {run["label"]: run for run in transcript["runs"]}


def _names(run: dict[str, Any]) -> list[str]:
    return [event["event"] for event in run["events"]]


def _text(run: dict[str, Any]) -> str:
    return "".join(e["data"]["text"] for e in run["events"] if e["event"] == "token")


def test_models_are_pinned_by_digest(transcript: dict[str, Any]) -> None:
    models = transcript["models"]
    assert models["generation"]["name"] == "qwen3:32b"
    assert models["embedding"]["name"].startswith("nomic-embed-text")
    for model in models.values():
        assert _DIGEST.match(model["digest"]), model


def test_every_run_is_tokens_then_optional_retraction_then_done(
    transcript: dict[str, Any],
) -> None:
    for run in transcript["runs"]:
        names = _names(run)
        assert names[-1] == "done"
        body = names[:-1]
        if body and body[-1] == "retracted":
            body = body[:-1]
        assert body, f"{run['label']}: no token before done"
        assert set(body) == {"token"}, f"{run['label']}: {names}"


def test_timestamps_are_monotonic(transcript: dict[str, Any]) -> None:
    for run in transcript["runs"]:
        stamps = [event["t_ms"] for event in run["events"]]
        assert stamps == sorted(stamps)
        assert stamps[0] >= 0


def test_done_carries_sources_and_the_documented_fields(transcript: dict[str, Any]) -> None:
    for run in transcript["runs"]:
        done = run["events"][-1]["data"]
        assert set(done) == _DONE_KEYS
        assert done["question"] == run["question"]
        assert done["sources"], f"{run['label']}: done without sources"
        assert done["grounded"] is True
        assert done["refused"] is False
        assert done["truncated"] is False
        assert "generation" in done["timing_ms"]  # the model was really called


def test_answered_run_is_cited_and_never_retracted(transcript: dict[str, Any]) -> None:
    run = _runs(transcript)["answered"]
    assert "retracted" not in _names(run)
    text = _text(run)
    n_chunks = run["k"]
    assert validate_output(text, max_citation=n_chunks).ok, text


def test_retraction_search_is_reported_as_it_happened(transcript: dict[str, Any]) -> None:
    search = transcript["retraction_search"]
    attempts = search["attempts"]
    assert 1 <= len(attempts) <= search["max_attempts"]
    assert [a["attempt"] for a in attempts] == list(range(1, len(attempts) + 1))
    for attempt in attempts:
        assert attempt["question"] in search["candidates"]
    retracted = [a for a in attempts if a["retracted"]]
    assert search["found"] is bool(retracted)
    if retracted:  # the search stops at the first retraction
        assert attempts[-1]["retracted"] is True
        assert len(retracted) == 1
    else:
        assert len(attempts) == search["max_attempts"]
        assert "retracted" not in _runs(transcript)


def test_retracted_run_replaces_ungrounded_text_with_the_abstention(
    transcript: dict[str, Any],
) -> None:
    runs = _runs(transcript)
    if not transcript["retraction_search"]["found"]:
        pytest.skip("no retraction observed within the declared attempts")
    run = runs["retracted"]
    names = _names(run)
    assert names.count("retracted") == 1
    assert names[-2:] == ["retracted", "done"]
    streamed = _text(run)
    assert not validate_output(streamed, max_citation=run["k"]).ok, streamed
    retraction = run["events"][-2]["data"]
    assert set(retraction) == {"reason", "answer"}
    assert str(retraction["reason"]).startswith("ungrounded")
    assert is_abstention(str(retraction["answer"]))
    assert run["question"] == transcript["retraction_search"]["attempts"][-1]["question"]

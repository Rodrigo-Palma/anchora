"""Integrity tests for the held-out evaluation set and the few-shot baseline.

These guard the methodology fix: the holdout must be genuinely disjoint from the
training golden set (no leakage), abstention cases must carry the exact refusal
sentence, and the few-shot exemplars must never be drawn from the holdout. All
offline and deterministic — no model, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

from anchora import metrics
from anchora.evals import load_cases as load_golden
from anchora.guardrails import _ABSTENTION
from anchora.rag import retrieve
from anchora.store import VectorStore

_ROOT = Path(__file__).resolve().parents[1]
_HOLDOUT_PATH = _ROOT / "data" / "golden" / "holdout.json"
sys.path.insert(0, str(_ROOT / "scripts"))

import evaluate_finetune as ef  # noqa: E402


def _holdout() -> list[dict[str, object]]:
    return ef.load_cases(_HOLDOUT_PATH)


def test_holdout_schema_and_size() -> None:
    cases = _holdout()
    assert len(cases) >= 20
    for case in cases:
        assert {"id", "question", "expected_doc", "answerable", "reference_answer"} <= set(case)


def test_holdout_has_answerable_and_abstention_cases() -> None:
    cases = _holdout()
    answerable = [c for c in cases if c["answerable"]]
    abstention = [c for c in cases if not c["answerable"]]
    assert len(answerable) >= 15
    assert len(abstention) >= 3


def test_abstention_cases_use_exact_refusal_sentence() -> None:
    for case in _holdout():
        if not case["answerable"]:
            assert _ABSTENTION in str(case["reference_answer"]).lower()
            assert case["expected_doc"] == "NONE"


def test_holdout_is_disjoint_from_training() -> None:
    """No question id or text may appear in both train and holdout (no leakage)."""
    train = load_golden()
    train_ids = {c["id"] for c in train}
    train_questions = {c["question"].strip().lower() for c in train}
    for case in _holdout():
        assert case["id"] not in train_ids
        assert str(case["question"]).strip().lower() not in train_questions


def test_fewshot_exemplars_are_not_in_holdout() -> None:
    holdout_ids = {c["id"] for c in _holdout()}
    assert not (set(ef._FEWSHOT_EXEMPLAR_IDS) & holdout_ids)


def test_fewshot_prefix_builds_offline(store: VectorStore) -> None:
    prefix = ef.build_fewshot_prefix(store)
    assert prefix.strip()
    # teaches both behaviors: a citation and the exact abstention sentence
    assert "[1]" in prefix
    assert _ABSTENTION in prefix.lower()


def test_holdout_retrieval_recall_floor(store: VectorStore) -> None:
    """Retrieval should still find most expected docs on unseen questions.

    A floor (not == 1.0) on purpose: the offline glossary bridge is fit to the
    training questions, so some generalization loss here is expected and honest.
    """
    answerable = [c for c in _holdout() if c["answerable"]]
    recalls = [
        metrics.context_recall(
            [chunk.doc_id for chunk in retrieve(store, str(c["question"]), k=4, provider="hash")],
            str(c["expected_doc"]),
        )
        for c in answerable
    ]
    assert sum(recalls) / len(recalls) >= 0.8

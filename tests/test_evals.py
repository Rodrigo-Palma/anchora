from __future__ import annotations

from anchora.config import settings
from anchora.evals import load_cases, run


def test_golden_set_loads() -> None:
    cases = load_cases()
    assert len(cases) >= 20
    for case in cases:
        assert {"id", "question", "expected_doc", "reference_answer"} <= set(case)


def test_eval_recall_is_perfect() -> None:
    """The deterministic hash retriever must surface every expected doc."""
    report = run()
    assert report.mean_recall == 1.0, [s.case_id for s in report.scores if s.recall < 1.0]


def test_eval_faithfulness_meets_threshold() -> None:
    report = run()
    assert report.mean_faithfulness >= settings.faithfulness_threshold

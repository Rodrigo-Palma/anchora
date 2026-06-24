"""Offline, reproducible evaluation harness (deterministic hash embeddings).

Loads the golden set, ingests the corpus, and for every case measures:

* retrieval — context precision / recall against the expected document;
* generation — faithfulness and answer relevance of the agent's answer
  (extractive offline answer, so the suite needs no model or network).

Runs in CI and gates the build: it exits non-zero if retrieval recall is not
perfect or if mean faithfulness drops below ``settings.faithfulness_threshold``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from anchora import metrics
from anchora.agent import Agent
from anchora.config import settings
from anchora.ingest import ingest_dir
from anchora.llm import build_context
from anchora.rag import retrieve

_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_GOLDEN_PATH = _ROOT / "data" / "golden" / "golden.json"

_PROVIDER = "hash"  # deterministic, offline


@dataclass
class CaseScore:
    case_id: str
    question: str
    expected_doc: str
    recall: float
    precision: float
    faithfulness: float
    answer_relevance: float


@dataclass
class Report:
    scores: list[CaseScore]

    @property
    def mean_recall(self) -> float:
        return _mean(s.recall for s in self.scores)

    @property
    def mean_precision(self) -> float:
        return _mean(s.precision for s in self.scores)

    @property
    def mean_faithfulness(self) -> float:
        return _mean(s.faithfulness for s in self.scores)

    @property
    def mean_answer_relevance(self) -> float:
        return _mean(s.answer_relevance for s in self.scores)


def load_cases(path: Path = _GOLDEN_PATH) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["cases"])


def run(k: int = 4) -> Report:
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    # use_llm=False keeps the eval fully offline and deterministic.
    agent = Agent(store, k=k, provider=_PROVIDER, use_llm=False)
    scores: list[CaseScore] = []
    for case in load_cases():
        question = case["question"]
        expected = case["expected_doc"]
        chunks = retrieve(store, question, k=k, provider=_PROVIDER)
        retrieved_docs = [c.doc_id for c in chunks]
        result = agent.run(question)
        context = build_context(chunks)
        scores.append(
            CaseScore(
                case_id=case["id"],
                question=question,
                expected_doc=expected,
                recall=metrics.context_recall(retrieved_docs, expected),
                precision=metrics.context_precision(retrieved_docs, expected),
                faithfulness=metrics.faithfulness(result.answer, context),
                answer_relevance=metrics.answer_relevance(result.answer, question),
            )
        )
    return Report(scores=scores)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 4) if items else 0.0


def main() -> None:
    report = run()
    print(f"{'case':<22} {'recall':>7} {'prec':>7} {'faith':>7} {'ans_rel':>8}")
    print("-" * 56)
    for s in report.scores:
        print(
            f"{s.case_id:<22} {s.recall:>7.2f} {s.precision:>7.2f} "
            f"{s.faithfulness:>7.2f} {s.answer_relevance:>8.2f}"
        )
    print("-" * 56)
    print(
        f"{'MEAN':<22} {report.mean_recall:>7.2f} {report.mean_precision:>7.2f} "
        f"{report.mean_faithfulness:>7.2f} {report.mean_answer_relevance:>8.2f}"
    )

    failures: list[str] = []
    if report.mean_recall < 1.0:
        misses = [s.case_id for s in report.scores if s.recall < 1.0]
        failures.append(f"retrieval recall {report.mean_recall:.2f} < 1.00 (missed: {misses})")
    if report.mean_faithfulness < settings.faithfulness_threshold:
        failures.append(
            f"faithfulness {report.mean_faithfulness:.2f} < {settings.faithfulness_threshold:.2f}"
        )

    if failures:
        print("\nEVAL GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nEVAL GATE PASSED")


if __name__ == "__main__":
    main()

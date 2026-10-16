"""Measure the out-of-domain floor signals on the production embedder.

The out-of-domain floor (:mod:`anchora.domain`) combines a lexical signal (the
number of distinct corpus tokens a question shares) and an optional dense one
(the top cosine to any corpus chunk). The dense one is only meaningful on the
real embedder, so this script works in two steps, like the judge calibration:

1. ``--run`` (needs Ollama with ``nomic-embed-text``) records, for every
   labelled question, its distinct overlap and its top nomic cosine, together
   with the Ollama version and the model digest, in
   ``data/eval/ood-similarities.json``.
2. With no flags (or ``--check`` in CI) it rebuilds the "abstains by floor"
   tables from that frozen file, with no model and no network, recomputes the
   lexical overlap from the corpus, and fails if either drifts from the tables
   published in ADR 6.

Labelled sets: golden (in-domain, n=24), holdout answerable (n=22), holdout
out-of-corpus (n=6) and the ``ood-*`` attacks (n=12).

Usage::

    uv run python scripts/calibrate_ood_floor.py --run     # needs Ollama
    uv run python scripts/calibrate_ood_floor.py           # tables from frozen file
    uv run python scripts/calibrate_ood_floor.py --check   # fail on drift
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from anchora.config import settings
from anchora.embeddings import embed_texts, tokenize
from anchora.ingest import ingest_dir
from anchora.store import VectorStore

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_GOLDEN_PATH = _ROOT / "data" / "golden" / "golden.json"
_HOLDOUT_PATH = _ROOT / "data" / "golden" / "holdout.json"
_ATTACKS_PATH = _ROOT / "data" / "adversarial" / "attacks.json"
_BENIGN_PATH = _ROOT / "data" / "adversarial" / "benign.json"
FROZEN_PATH = _ROOT / "data" / "eval" / "ood-similarities.json"
ADR_PATH = _ROOT / "docs" / "adr" / "0006-out-of-domain-floor.md"

SUBSETS = ("golden", "holdout_answerable", "holdout_out_of_corpus", "ood")
_HEADERS = (
    "golden in-domain (n=24)",
    "holdout answerable (n=22)",
    "holdout out-of-corpus (n=6)",
    "`ood-*` attacks (n=12)",
)
CURRENT_MIN_OVERLAP = 2
OVERLAP_FLOORS = (1, 2, 3)
# Fixed grid, chosen before the cosines were measured; 0.0 is the shipped default.
COSINE_THRESHOLDS = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
_HTTP_TIMEOUT_S = 30.0

SimilarityFn = Callable[[str], float]


@dataclass(frozen=True)
class Question:
    case_id: str
    subset: str
    text: str
    pinned: bool


def _cases(path: Path, key: str = "cases") -> list[dict[str, Any]]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = data[key]
    return cases


def collect_questions() -> list[Question]:
    """Every labelled question, tagged with its subset and pinned over-block flag."""
    benign = json.loads(_BENIGN_PATH.read_text(encoding="utf-8"))
    pinned = set(benign.get("known_over_blocks", {}))
    questions = [Question(c["id"], "golden", c["question"], False) for c in _cases(_GOLDEN_PATH)]
    for case in _cases(_HOLDOUT_PATH):
        subset = "holdout_answerable" if case["answerable"] else "holdout_out_of_corpus"
        questions.append(Question(case["id"], subset, case["question"], case["id"] in pinned))
    questions.extend(
        Question(a["id"], "ood", a["input"], False)
        for a in _cases(_ATTACKS_PATH, "attacks")
        if a["category"] == "off_domain"
    )
    return questions


def distinct_overlap(question: str, store: VectorStore) -> int:
    """Same signal as :func:`anchora.domain.assess_domain` uses."""
    return len(set(tokenize(question, query=True)) & store.corpus_vocabulary())


def measure(
    questions: Sequence[Question], store: VectorStore, similarity: SimilarityFn
) -> list[dict[str, Any]]:
    """One record per question: its lexical overlap and its top cosine."""
    return [
        {
            "case_id": q.case_id,
            "subset": q.subset,
            "pinned": q.pinned,
            "distinct_overlap": distinct_overlap(q.text, store),
            "top_cosine": round(similarity(q.text), 6),
        }
        for q in questions
    ]


def _abstain_counts(
    records: Sequence[dict[str, Any]], min_overlap: int, threshold: float
) -> tuple[int, ...]:
    """Abstentions per subset under the wired AND rule (overlap, then cosine)."""

    def abstains(r: dict[str, Any]) -> bool:
        return bool(r["distinct_overlap"] < min_overlap or r["top_cosine"] < threshold)

    return tuple(sum(1 for r in records if r["subset"] == s and abstains(r)) for s in SUBSETS)


def overlap_rows(records: Sequence[dict[str, Any]]) -> list[tuple[int, tuple[int, ...]]]:
    return [(floor, _abstain_counts(records, floor, 0.0)) for floor in OVERLAP_FLOORS]


def cosine_rows(records: Sequence[dict[str, Any]]) -> list[tuple[float, tuple[int, ...]]]:
    return [(t, _abstain_counts(records, CURRENT_MIN_OVERLAP, t)) for t in COSINE_THRESHOLDS]


def _table(first: str, rows: Sequence[tuple[str, Sequence[int | str]]]) -> str:
    lines = [
        f"| {first} | " + " | ".join(_HEADERS) + " |",
        "|---" * (len(_HEADERS) + 1) + "|",
    ]
    lines.extend(
        f"| {label} | " + " | ".join(str(v) for v in values) + " |" for label, values in rows
    )
    return "\n".join(lines)


def _range(records: Sequence[dict[str, Any]], subset: str) -> str:
    values = [r["top_cosine"] for r in records if r["subset"] == subset]
    return f"{min(values):.3f} to {max(values):.3f}"


def render_tables(records: Sequence[dict[str, Any]]) -> list[str]:
    """The markdown tables ADR 6 publishes, rebuilt from the frozen records."""
    overlap = _table("Overlap floor (cosine off)", [(str(f), c) for f, c in overlap_rows(records)])
    cosine = _table(
        f"Cosine floor (overlap ≥ {CURRENT_MIN_OVERLAP})",
        [(f"{t:.2f}", c) for t, c in cosine_rows(records)],
    )
    ranges = _table("Top nomic cosine", [("range", [_range(records, s) for s in SUBSETS])])
    pinned_lines = [
        "| Pinned holdout question | distinct overlap | top nomic cosine |",
        "|---|---|---|",
    ]
    pinned_lines.extend(
        f"| `{r['case_id']}` | {r['distinct_overlap']} | {r['top_cosine']:.3f} |"
        for r in records
        if r["pinned"]
    )
    return [overlap, cosine, ranges, "\n".join(pinned_lines)]


def model_identity(model: str, client: httpx.Client | None = None) -> dict[str, Any]:
    """Ollama version, model digest and details (what ``ollama show`` reports)."""
    http = client or httpx.Client(base_url=settings.ollama_base_url, timeout=_HTTP_TIMEOUT_S)
    tags = http.get("/api/tags").json().get("models", [])
    names = {model, f"{model}:latest"}
    match = next((m for m in tags if m.get("name") in names), None)
    if match is None:
        raise RuntimeError(f"{model} is not pulled in Ollama (ollama pull {model})")
    version = http.get("/api/version").json().get("version", "unknown")
    details = http.post("/api/show", json={"model": model}).json().get("details", {})
    return {
        "name": match["name"],
        "digest": match["digest"],
        "ollama_version": version,
        "details": details,
    }


def _ollama_similarity(store: VectorStore) -> SimilarityFn:
    def similarity(question: str) -> float:
        vector = embed_texts([question], provider="ollama", query=True)[0]
        hits = store.search_indices(vector, k=1)
        return hits[0][1] if hits else 0.0

    return similarity


def run(path: Path = FROZEN_PATH) -> dict[str, Any]:
    """Measure on the production embedder and freeze the records (needs Ollama)."""
    before = model_identity(settings.embed_model)
    store = ingest_dir(_CORPUS_DIR, provider="ollama")
    records = measure(collect_questions(), store, _ollama_similarity(store))
    after = model_identity(settings.embed_model)
    if before["digest"] != after["digest"]:
        raise RuntimeError("model digest changed during the run; re-run")
    frozen = {
        "_comment": (
            "Distinct lexical overlap and top nomic-embed-text cosine per labelled "
            "question. Regenerate with scripts/calibrate_ood_floor.py --run; "
            "--check rebuilds the ADR 6 tables offline."
        ),
        "measured_at": datetime.now(UTC).date().isoformat(),
        "model": before,
        "chunks": len(store),
        "records": records,
    }
    path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return frozen


def load_frozen(path: Path = FROZEN_PATH) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def check(frozen_path: Path = FROZEN_PATH, adr_path: Path = ADR_PATH) -> list[str]:
    """Drift between the frozen file, the corpus overlap and the published tables."""
    records = load_frozen(frozen_path)["records"]
    failures: list[str] = []
    expected = {q.case_id: q for q in collect_questions()}
    if {r["case_id"] for r in records} != set(expected):
        failures.append("question set drift: frozen ids differ from the labelled sets")
    store = ingest_dir(_CORPUS_DIR, provider="hash")
    for r in records:
        q = expected.get(r["case_id"])
        if q is None:
            continue
        if distinct_overlap(q.text, store) != r["distinct_overlap"]:
            failures.append(f"overlap drift: {r['case_id']}")
        if not math.isfinite(r["top_cosine"]):
            failures.append(f"non-finite cosine: {r['case_id']}")
    adr = adr_path.read_text(encoding="utf-8")
    failures.extend(
        f"table not published in {adr_path.name}:\n{table}"
        for table in render_tables(records)
        if table not in adr
    )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true", help="measure with Ollama and freeze")
    group.add_argument("--check", action="store_true", help="fail on drift (CI)")
    args = parser.parse_args(argv)

    if args.run:
        frozen = run()
        print(f"froze {len(frozen['records'])} records -> {FROZEN_PATH.relative_to(_ROOT)}")
    records = load_frozen()["records"]
    for table in render_tables(records):
        print(table, end="\n\n")
    if args.check:
        failures = check()
        for failure in failures:
            print(f"FAIL {failure}")
        if failures:
            return 1
        print("ood floor: frozen similarities match ADR 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

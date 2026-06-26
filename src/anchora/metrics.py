"""Deterministic, offline proxies for RAG quality metrics.

DeepEval / RAGAS compute these with an LLM judge. That is great locally but is
non-deterministic and (with hosted judges) costs money — unfit for a CI gate.
These lexical proxies are honest, reproducible stand-ins that gate CI for free;
the real LLM-judge versions can be run locally via the Ollama judge (see
``scripts/compare_evals.py``).

All metrics return a float in [0, 1].
"""

from __future__ import annotations

import re
import unicodedata

from anchora.embeddings import bridge_tokens

# fmt: off
_STOPWORDS = frozenset({
    "a", "o", "e", "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "um", "uma", "que", "para", "por", "com", "sem", "como", "qual", "quais",
    "quando", "onde", "quem", "ao", "aos", "as", "os", "é", "ou", "se", "sua",
    "seu", "suas", "seus",
    # English (questions are in English; see anchora.embeddings._GLOSSARY)
    "the", "of", "to", "in", "is", "are", "what", "who", "how", "does", "did",
    "for", "under", "new", "long", "many", "their", "after", "once", "with",
})
# fmt: on


def tokenize(text: str) -> set[str]:
    """Accent-folded, lowercased content tokens (stopwords removed).

    English terms are bridged to their Portuguese corpus equivalents so an
    English question can be scored against a Portuguese answer/context (the same
    cross-lingual bridge used for retrieval).
    """
    folded = "".join(
        ch for ch in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(ch)
    )
    words = re.findall(r"[a-z0-9]+", folded)
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    return {w for w in bridge_tokens(content) if len(w) > 2}


def _coverage(target: set[str], source: set[str]) -> float:
    """Fraction of ``target`` tokens that appear in ``source``."""
    if not target:
        return 0.0
    return len(target & source) / len(target)


def faithfulness(answer: str, context: str) -> float:
    """How much of the answer is supported by the retrieved context.

    Citation markers like ``[1]`` are stripped before scoring.
    """
    clean = re.sub(r"\[\d+\]", " ", answer)
    return round(_coverage(tokenize(clean), tokenize(context)), 4)


def answer_relevance(answer: str, question: str) -> float:
    """How much of the question's intent the answer addresses."""
    return round(_coverage(tokenize(question), tokenize(answer)), 4)


def context_precision(retrieved_docs: list[str], expected_doc: str) -> float:
    """Fraction of retrieved chunks that come from the expected document."""
    if not retrieved_docs:
        return 0.0
    hits = sum(1 for doc in retrieved_docs if doc == expected_doc)
    return round(hits / len(retrieved_docs), 4)


def context_recall(retrieved_docs: list[str], expected_doc: str) -> float:
    """1.0 if the expected document was retrieved at all, else 0.0."""
    return 1.0 if expected_doc in retrieved_docs else 0.0


_CITATION_INDEX_RE = re.compile(r"\[(\d+)\]")


def citation_correct(answer: str, retrieved_docs: list[str], expected_doc: str) -> float:
    """Whether the answer cites the *right* document, not just any bracket.

    Resolves each ``[n]`` marker to the n-th retrieved chunk and returns 1.0 if at
    least one cited index points at ``expected_doc``. This is the honest grounding
    signal: ``guardrails.validate_output`` only checks that a bracket is present,
    so a model that learns to always append ``[1]`` scores "grounded" for free —
    this metric does not reward that.
    """
    for marker in _CITATION_INDEX_RE.findall(answer):
        index = int(marker)
        if 1 <= index <= len(retrieved_docs) and retrieved_docs[index - 1] == expected_doc:
            return 1.0
    return 0.0

from __future__ import annotations

from anchora.metrics import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
    tokenize,
)


def test_tokenize_folds_accents_and_drops_stopwords() -> None:
    toks = tokenize("O prazo para recurso é de 10 dias úteis")
    assert "prazo" in toks
    assert "uteis" in toks
    assert "para" not in toks  # stopword


def test_faithfulness_full_support() -> None:
    answer = "O prazo de recurso é de 10 dias [1]."
    context = "[1] O prazo de recurso é de 10 dias contados da ciência."
    assert faithfulness(answer, context) == 1.0


def test_faithfulness_partial() -> None:
    answer = "O prazo é elefante roxo voador."
    context = "[1] O prazo de recurso é de 10 dias."
    assert faithfulness(answer, context) < 1.0


def test_answer_relevance() -> None:
    # English question vs Portuguese answer exercises the cross-lingual bridge,
    # mirroring the real flow (English golden questions, Portuguese corpus).
    question = "What is the appeal deadline?"
    answer = "O prazo de recurso é de 10 dias."
    assert answer_relevance(answer, question) > 0.0


def test_context_precision() -> None:
    assert context_precision(["a.md", "a.md", "b.md", "c.md"], "a.md") == 0.5
    assert context_precision([], "a.md") == 0.0


def test_context_recall() -> None:
    assert context_recall(["a.md", "b.md"], "a.md") == 1.0
    assert context_recall(["b.md", "c.md"], "a.md") == 0.0


def test_citation_correct_rewards_right_document() -> None:
    from anchora.metrics import citation_correct

    docs = ["lai.md", "lei-8112.md", "lgpd.md", "cpc-prazos.md"]
    # cites [2] which resolves to lei-8112.md
    assert citation_correct("30 dias. [2]", docs, "lei-8112.md") == 1.0
    # has a bracket, but the cited index points at the wrong document
    assert citation_correct("30 dias. [1]", docs, "lei-8112.md") == 0.0
    # no citation at all
    assert citation_correct("30 dias.", docs, "lei-8112.md") == 0.0
    # out-of-range index does not crash or falsely match
    assert citation_correct("30 dias. [9]", docs, "lei-8112.md") == 0.0

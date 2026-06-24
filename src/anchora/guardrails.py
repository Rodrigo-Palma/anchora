"""Production-style guardrails for the RAG agent.

Three layers, all deterministic (no model needed, so they run in CI):

* input  — block prompt-injection / jailbreak attempts before retrieval;
* PII    — detect and redact Brazilian PII (CPF, e-mail, phone) in text;
* output — a grounded answer must cite a source or explicitly abstain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- output grounding -------------------------------------------------------

_CITATION_RE = re.compile(r"\[\d+\]")
_ABSTENTION = "could not find"


@dataclass
class GuardrailResult:
    ok: bool
    reason: str


def validate_output(answer: str) -> GuardrailResult:
    """Pass if the answer cites at least one source or explicitly abstains."""
    if _CITATION_RE.search(answer):
        return GuardrailResult(True, "cited")
    if _ABSTENTION in answer.lower():
        return GuardrailResult(True, "abstained")
    return GuardrailResult(False, "ungrounded: no citation and no abstention")


# --- input safety -----------------------------------------------------------

_INJECTION_PATTERNS = (
    r"ignore (all |as |todas as |todas |suas |the )?(instru|previous|above|prior)",
    r"disregard (the |all |previous |prior )?(instruc|rules|context)",
    r"esque(ç|c)a (as |suas |todas )?(instru|regras)",
    r"reveal (your |the )?(system )?prompt",
    r"(mostre|revele|repita) (o |seu )?(system )?prompt",
    r"you are now",
    r"voc(ê|e) agora (é|e) ",
    r"act as (an? )?(dan|jailbreak|unrestricted)",
    r"developer mode",
    r"sem (filtros|restri(ç|c)(õ|o)es|limites)",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def check_input(text: str) -> GuardrailResult:
    """Reject likely prompt-injection / jailbreak inputs."""
    if _INJECTION_RE.search(text):
        return GuardrailResult(False, "blocked: possible prompt injection / jailbreak")
    return GuardrailResult(True, "clean")


# --- PII detection / redaction ---------------------------------------------

_CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b")

_PII_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("CPF", _CPF_RE),
    ("EMAIL", _EMAIL_RE),
    ("PHONE", _PHONE_RE),
)


def detect_pii(text: str) -> list[str]:
    """Return the kinds of PII present in ``text`` (e.g. ``["CPF", "EMAIL"]``)."""
    return [label for label, pattern in _PII_RULES if pattern.search(text)]


def redact_pii(text: str) -> str:
    """Replace detected PII spans with ``[REDACTED_<KIND>]`` placeholders."""
    redacted = text
    for label, pattern in _PII_RULES:
        redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted

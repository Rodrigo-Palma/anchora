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

_CITATION_INDEX_RE = re.compile(r"\[(\d+)\]")
_ABSTENTION = "could not find"

# Conservative Portuguese refusal markers. The adapter sometimes declines in
# Portuguese instead of emitting the canonical English sentence, and exact-string
# matching scored those real refusals as failures. These phrasings are explicit
# enough not to collide with a factual legal answer. The proper long-term fix is
# to constrain the model to the canonical sentence — this just stops the metric
# from under-counting the obvious cases meanwhile.
_PT_ABSTENTION = (
    "nenhum dado foi fornecido",
    "não há uma data específica",
    "não há informações",
    "não foi possível encontrar",
    "não consta",
)


@dataclass
class GuardrailResult:
    ok: bool
    reason: str


def is_abstention(answer: str) -> bool:
    """True if the answer explicitly declines to answer (English or Portuguese)."""
    lowered = answer.lower()
    if _ABSTENTION in lowered:
        return True
    return any(marker in lowered for marker in _PT_ABSTENTION)


def validate_output(answer: str, max_citation: int | None = None) -> GuardrailResult:
    """Pass if the answer cites at least one source or explicitly abstains.

    With ``max_citation`` set, every ``[n]`` must resolve to a retrieved chunk
    (``1 <= n <= max_citation``). A model coaxed into fabricating ``[99]`` — or
    into citing anything when nothing was retrieved — fails as ungrounded
    instead of passing on bracket presence alone.
    """
    markers = [int(m) for m in _CITATION_INDEX_RE.findall(answer)]
    if markers:
        if max_citation is not None and any(not 1 <= n <= max_citation for n in markers):
            bad = [n for n in markers if not 1 <= n <= max_citation]
            return GuardrailResult(False, f"ungrounded: citation out of range {bad}")
        return GuardrailResult(True, "cited")
    if is_abstention(answer):
        return GuardrailResult(True, "abstained")
    return GuardrailResult(False, "ungrounded: no citation and no abstention")


# --- input safety -----------------------------------------------------------

_INJECTION_PATTERNS = (
    r"ignore (all |as |todas as |todas |suas |the )?(instru|previous|above|prior)",
    r"disregard (the |all |previous |prior )?(instruc|rules|context)",
    r"(forget|discard) (your |the |all |previous |prior )*(instruc|rules|guidelines|context)",
    r"esque(ç|c)a (as |suas |todas )?(instru|regras)",
    r"(reveal|show|print|expose|output|leak|repeat) (me |back )?(your |the |everything )?"
    r"(system |initial |original )?(prompt|instruc|rules)",
    r"(mostre|revele|repita|imprima|exiba) (o |seu |a |as |tudo )?(system )?(prompt|instru|regras)",
    r"repeat (everything|all|the text) (above|before|prior)",
    r"you are now",
    r"voc(ê|e) agora (é|e) ",
    r"act as (an? )?(dan|jailbreak|unrestricted|assistant with no)",
    r"pretend (you |to )?(have no|are|there are no)",
    r"(finja|finge) (que |ter )?(n(ã|a)o|sem)",
    r"(override|bypass|skip|turn off|disable) (your |the |all )?(safety|content|filter|rule|guard)",
    r"(if you had|without any|com nenhum|sem nenhum) (no )?(guidelines|rules|regras|restri)",
    r"developer mode",
    r"^system:",
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

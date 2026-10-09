"""A small tool-using RAG agent with guardrails.

Flow per question:

1. input guardrail — reject prompt-injection / jailbreak;
2. plan — decide which tools to call (heuristic, deterministic by default);
3. act — run tools, retrieve grounding chunks;
4. answer — generate a cited answer with the local model, or an extractive
   fallback when the model is offline (keeps the agent testable);
5. output guardrail — require a citation or an explicit abstention.

The planner is deliberately rule-based so the whole agent is reproducible in
CI without a model. A model-backed planner can be layered on later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from anchora import guardrails
from anchora.domain import assess_domain
from anchora.llm import answer as llm_answer
from anchora.observability import Trace
from anchora.rag import retrieve
from anchora.store import Chunk, VectorStore
from anchora.tools import legal_deadline

_REFUSAL = "I cannot fulfill this request for security reasons."
_NOT_FOUND = "I could not find this information in the provided documents."

# A "DD/MM/YYYY" or "YYYY-MM-DD" date plus an "N days" span signals a deadline question.
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b|\b(\d{2})/(\d{2})/(\d{4})\b")
_DAYS_RE = re.compile(r"\b(\d{1,3})\s*(?:calendar|business)?\s*days?\b", re.IGNORECASE)
_CALENDAR_RE = re.compile(r"calendar", re.IGNORECASE)


@dataclass
class ToolCall:
    name: str
    args: dict[str, object]
    output: str


@dataclass
class AgentResult:
    question: str
    answer: str
    sources: list[str]
    grounded: bool
    tool_calls: list[ToolCall] = field(default_factory=list)
    refused: bool = False
    trace: Trace = field(default_factory=Trace)
    # Why the output guardrail replaced the model's answer, or None if it did not.
    retraction: str | None = None


@dataclass
class PreparedQuestion:
    """Everything known about a question before the answer is generated."""

    question: str
    trace: Trace
    chunks: list[Chunk] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    deadline_fact: str | None = None
    result: AgentResult | None = None  # set when no generation is needed


class Agent:
    def __init__(
        self,
        store: VectorStore,
        k: int = 4,
        provider: str | None = None,
        use_llm: bool = True,
    ) -> None:
        self._store = store
        self._k = k
        self._provider = provider
        self._use_llm = use_llm

    def run(self, question: str) -> AgentResult:
        prepared = self.prepare(question)
        if prepared.result is not None:
            return prepared.result
        with prepared.trace.stage("generation"):
            answer_text = self._compose_answer(question, prepared.chunks, prepared.deadline_fact)
        return self.finalize(prepared, answer_text)

    def prepare(self, question: str) -> PreparedQuestion:
        """Run every step before generation: input guardrail, tools, domain, retrieval.

        Returns a :class:`PreparedQuestion` whose ``result`` is already final when
        the question was refused or is out of domain, so a streaming caller can
        stop before calling the model.
        """
        trace = Trace()
        with trace.stage("guardrail_input"):
            guard = guardrails.check_input(question)
        if not guard.ok:
            refused = AgentResult(
                question=question,
                answer=_REFUSAL,
                sources=[],
                grounded=False,
                refused=True,
                trace=trace,
            )
            return PreparedQuestion(question, trace, result=refused)

        tool_calls: list[ToolCall] = []

        # Optional deadline computation when the question carries a date + span.
        deadline_fact = self._maybe_compute_deadline(question, tool_calls)

        # Out-of-domain floor: a question that barely touches the corpus cannot
        # be grounded, so abstain instead of quoting the nearest-by-cosine chunk
        # with an irrelevant citation. The floor requires several distinct
        # corpus tokens (not just one incidental collision); see
        # anchora.domain. Deadline-tool questions still get their computed fact
        # appended below.
        with trace.stage("domain_check"):
            verdict = assess_domain(question, self._store, provider=self._provider)
        if not verdict.in_domain:
            abstained = AgentResult(
                question=question,
                answer=_append_fact(_NOT_FOUND, deadline_fact),
                sources=[],
                grounded=True,  # an explicit abstention is itself grounded
                tool_calls=tool_calls,
                trace=trace,
            )
            return PreparedQuestion(question, trace, result=abstained)

        # Always ground answers in retrieved chunks.
        with trace.stage("retrieval"):
            chunks = retrieve(self._store, question, k=self._k, provider=self._provider)
        tool_calls.append(
            ToolCall(
                name="search_documents",
                args={"query": question, "k": self._k},
                output=f"{len(chunks)} excerpts retrieved.",
            )
        )
        return PreparedQuestion(question, trace, chunks, tool_calls, deadline_fact)

    def finalize(self, prepared: PreparedQuestion, answer_text: str) -> AgentResult:
        """Apply the output guardrail to a complete answer; abstain if ungrounded."""
        with prepared.trace.stage("guardrail_output"):
            verdict = guardrails.validate_output(answer_text, max_citation=len(prepared.chunks))
        final = answer_text if verdict.ok else _NOT_FOUND
        return AgentResult(
            question=prepared.question,
            answer=final,
            sources=sorted({chunk.title or chunk.doc_id for chunk in prepared.chunks}),
            grounded=True,  # either cited, or an explicit abstention (itself grounded)
            tool_calls=prepared.tool_calls,
            trace=prepared.trace,
            retraction=None if verdict.ok else verdict.reason,
        )

    def extractive_answer(self, prepared: PreparedQuestion) -> str:
        """The deterministic offline answer for a prepared question."""
        if not prepared.chunks:
            return _NOT_FOUND
        return _append_fact(_extractive_answer(prepared.chunks), prepared.deadline_fact)

    def _maybe_compute_deadline(self, question: str, tool_calls: list[ToolCall]) -> str | None:
        date_match = _DATE_RE.search(question)
        days_match = _DAYS_RE.search(question)
        if not (date_match and days_match):
            return None
        iso = _normalize_date(date_match)
        days = int(days_match.group(1))
        kind = "calendar" if _CALENDAR_RE.search(question) else "business"
        try:
            fact = legal_deadline(iso, days, kind)
        except ValueError:
            return None
        tool_calls.append(
            ToolCall(
                name="legal_deadline",
                args={"start_date": iso, "days": days, "kind": kind},
                output=fact,
            )
        )
        return fact

    def _compose_answer(self, question: str, chunks: list[Chunk], deadline_fact: str | None) -> str:
        if not chunks:
            return _NOT_FOUND
        if self._use_llm:
            generated = llm_answer(question, chunks)
            if generated:
                return _append_fact(generated, deadline_fact)
        return _append_fact(_extractive_answer(chunks), deadline_fact)


def _extractive_answer(chunks: list[Chunk]) -> str:
    """Deterministic offline fallback: quote the top chunk with a citation."""
    top = chunks[0]
    snippet = _first_sentences(top.text, 2)
    return f"According to {top.title or top.doc_id} [1]: {snippet}"


def _append_fact(answer: str, fact: str | None) -> str:
    if not fact:
        return answer
    return f"{answer}\n\n{fact}"


def _first_sentences(text: str, n: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def _normalize_date(match: re.Match[str]) -> str:
    if match.group(1):
        return match.group(1)
    return f"{match.group(4)}-{match.group(3)}-{match.group(2)}"

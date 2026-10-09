# 8. Stream real tokens, validate at the end, retract if ungrounded

Date: 2026-10-09 · Status: Accepted

## Context

`POST /ask/stream` used to compose the full answer (generation and the output
guardrail) and then split it into words. It had the SSE framing but no latency
benefit: the first byte arrived after the whole generation.

Streaming real tokens from Ollama (`stream: true`) moves the first token to the
start of decoding, but it breaks an assumption of the output guardrail. The
grounding check (`validate_output`: at least one `[n]` and every index resolving
to a retrieved chunk, or an explicit abstention) needs the complete text. A
citation usually comes at the end of a sentence, so no prefix of the answer can
be judged. By the time the check can run, the client has already shown the text.

Options considered:

| Option | First token | Ungrounded text reaches the client? |
|---|---|---|
| Buffer, validate, then stream (the old behaviour) | after full generation | never |
| Stream and validate each sentence | after first sentence | yes, every sentence before the failing one |
| **Stream, validate at the end, retract** | at the start of decoding | yes, until the `retracted` event |

## Decision

Stream tokens as they decode and run the output guardrail on the complete text
when the model finishes. If the text is not grounded, emit a `retracted` event
(`reason`, and the abstention that replaces the answer) before the terminal
`done` event. A client must render the stream as provisional and replace it on
`retracted`; the non-streaming `POST /ask` keeps the buffer-then-validate path
for callers that cannot do that.

The steps that can refuse without the model still run first: the input
guardrail, the out-of-domain floor and retrieval happen before any token is
requested, so a refused or out-of-domain question never reaches the model and
never streams model text. If Ollama is unreachable before the first token, the
endpoint falls back to the extractive answer, which goes through the same final
check. If the stream breaks midway, the partial text is validated like a
complete one.

## Consequences

- Time to first token drops to the model's own first-token latency.
- Ungrounded text can be on screen for the length of one generation. The
  guarantee moves from "never shown" to "always retracted before `done`", and
  the client contract has to say so. Tests (`tests/test_streaming.py`, with a
  mock Ollama transport) check that tokens reach the caller before decoding
  ends, that an answer with no citation or a forged `[99]` ends in `retracted`,
  and that a refused question never calls the model.
- The guardrail is still the deterministic citation check. It does not detect
  a claim that cites a real chunk but misstates it; that is what the judge
  calibration in `docs/eval-calibration.md` measures, offline.

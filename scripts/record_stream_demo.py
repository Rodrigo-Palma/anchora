"""Record the local-model streaming demo into a frozen SSE transcript.

Run against a live server backed by Ollama (``uv run anchora serve``):

    uv run python scripts/record_stream_demo.py --base-url http://127.0.0.1:8000

It streams one golden question that should be answered with a citation, then
searches for a retraction (ADR 8): it walks the pre-registered ``CANDIDATES``
in order, one ``POST /ask/stream`` per attempt, and stops at the first
``retracted`` event or after ``--max-attempts``. Every attempt is written down,
including the ones that did not retract. Tokens are echoed to the terminal as
they arrive, so recording this script with asciinema captures the same run
that ends up in ``data/demo/stream-transcript.json``.

The model is not deterministic; the transcript is an observed example, not a
guarantee. ``tests/test_demo_transcript.py`` checks its order and schema offline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from anchora.config import settings

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "data" / "demo" / "stream-transcript.json"

ANSWERED_QUESTION = (
    "What is the deadline for a public agency to respond to an access to information request?"
)

# Pre-registered before any recording run. Each asks, in plain user terms, for a
# format that leaves no room for an inline [n] (a bare number, yes/no, a single
# word). In-domain on purpose: an out-of-domain question is refused before the
# model is called and can never retract.
CANDIDATES = (
    "Answer with just a number: how many days does a public agency have to respond "
    "to an access to information request?",
    "Answer only yes or no: are procedural time limits in days counted in business "
    "days under the new CPC?",
    "In one word, which bidding modality was created by the new Procurement Law 14.133?",
    "Reply with only the number of years: when does the Administration's right to "
    "annul acts favorable to the citizen lapse under Law 9.784?",
    "Just the number, no explanation: how many days is the probationary period of a "
    "federal civil servant under Law 8.112?",
)

DEFAULT_MAX_ATTEMPTS = 10


def _model_digests(client: httpx.Client) -> dict[str, dict[str, str]]:
    tags = client.get(f"{settings.ollama_base_url}/api/tags").json()["models"]
    by_name = {model["name"]: model["digest"] for model in tags}

    def pick(name: str) -> dict[str, str]:
        for candidate in (name, f"{name}:latest"):
            if candidate in by_name:
                return {"name": candidate, "digest": by_name[candidate]}
        raise SystemExit(f"model {name!r} is not installed in Ollama")

    return {"generation": pick(settings.gen_model), "embedding": pick(settings.embed_model)}


def _parse_frame(frame: str) -> tuple[str, dict[str, Any]]:
    name, data = frame.split("\n", 1)
    return name.removeprefix("event: "), json.loads(data.removeprefix("data: "))


def _echo(name: str, data: dict[str, Any]) -> None:
    if name == "token":
        print(data["text"], end="", flush=True)
    elif name == "retracted":
        print(f"\n\n[retracted] reason={data['reason']!r}\n  -> {data['answer']}", flush=True)
    elif name == "done":
        gen_ms = data["timing_ms"].get("generation", 0.0)
        print(f"\n[done] sources={len(data['sources'])} generation={gen_ms / 1000:.1f}s\n")


def stream_once(client: httpx.Client, base_url: str, question: str, k: int) -> dict[str, Any]:
    """POST one question to /ask/stream and return its timed events."""
    print(f"$ ask/stream: {question}\n", flush=True)
    events: list[dict[str, Any]] = []
    start = time.perf_counter()
    buffer = ""
    payload = {"question": question, "k": k}
    with client.stream("POST", f"{base_url}/ask/stream", json=payload) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                name, data = _parse_frame(frame)
                t_ms = round((time.perf_counter() - start) * 1000, 1)
                events.append({"t_ms": t_ms, "event": name, "data": data})
                _echo(name, data)
    return {"question": question, "k": k, "events": events}


def _is_retracted(run: dict[str, Any]) -> bool:
    return any(event["event"] == "retracted" for event in run["events"])


def _search_retraction(
    client: httpx.Client, base_url: str, k: int, max_attempts: int
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    attempts: list[dict[str, Any]] = []
    for n in range(1, max_attempts + 1):
        question = CANDIDATES[(n - 1) % len(CANDIDATES)]
        run = stream_once(client, base_url, question, k)
        retractions = [e["data"]["reason"] for e in run["events"] if e["event"] == "retracted"]
        retracted = bool(retractions)
        reason = retractions[0] if retractions else None
        attempts.append(
            {"attempt": n, "question": question, "retracted": retracted, "reason": reason}
        )
        if retracted:
            return attempts, run
    return attempts, None


def record(base_url: str, max_attempts: int, out: Path) -> int:
    k = settings.top_k
    with httpx.Client(timeout=settings.request_timeout) as client:
        models = _model_digests(client)
        answered = stream_once(client, base_url, ANSWERED_QUESTION, k)
        if _is_retracted(answered):
            print("the golden question retracted; not recording it as 'answered'", file=sys.stderr)
            return 1
        attempts, retracted = _search_retraction(client, base_url, k, max_attempts)
    runs = [{"label": "answered", **answered}]
    if retracted is not None:
        runs.append({"label": "retracted", **retracted})
    transcript = {
        "description": "Observed SSE transcript of POST /ask/stream on local models "
        "(ADR 8). An observed example, not guaranteed behaviour: the model is not "
        "deterministic.",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "recorder": "scripts/record_stream_demo.py",
        "models": models,
        "retraction_search": {
            "max_attempts": max_attempts,
            "candidates": list(CANDIDATES),
            "attempts": attempts,
            "found": retracted is not None,
        },
        "runs": runs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(transcript, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    found = "found" if retracted is not None else "NOT found"
    print(f"retraction {found} in {len(attempts)}/{max_attempts} attempts -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--out", type=Path, default=_OUT)
    args = parser.parse_args(argv)
    return record(args.base_url.rstrip("/"), args.max_attempts, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

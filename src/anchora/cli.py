"""Command-line interface for anchora.

Subcommands:

* ``ingest``  — build a vector store from a corpus directory and save it;
* ``ask``     — answer a single question with the agent;
* ``eval``    — run the offline evaluation gate over the golden set;
* ``serve``   — start the FastAPI server (uvicorn).

Designed to work fully offline with ``--provider hash``; the default provider
uses the local Ollama models configured in ``config.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anchora.agent import Agent
from anchora.config import settings
from anchora.ingest import ingest_dir
from anchora.store import VectorStore

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CORPUS = _ROOT / "data" / "corpus"


def _cmd_ingest(args: argparse.Namespace) -> int:
    store = ingest_dir(args.corpus, provider=args.provider)
    if args.out:
        store.save(args.out)
        print(f"Indexed {len(store)} excerpts → {args.out}")
    else:
        print(f"Indexed {len(store)} excerpts (not saved; use --out to persist)")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    if args.store and Path(args.store).is_file():
        store = VectorStore.load(args.store)
    else:
        store = ingest_dir(args.corpus, provider=args.provider)
    agent = Agent(store, k=args.k, provider=args.provider, use_llm=not args.no_llm)
    result = agent.run(args.question)
    print(result.answer)
    if result.sources:
        print("\nSources: " + ", ".join(result.sources))
    if args.verbose:
        for tc in result.tool_calls:
            print(f"  [tool] {tc.name}: {tc.output}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from anchora.evals import main as eval_main

    eval_main()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("anchora.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anchora", description="Local-first legal RAG agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a corpus directory.")
    p_ingest.add_argument("--corpus", default=str(_DEFAULT_CORPUS))
    p_ingest.add_argument("--out", default="")
    p_ingest.add_argument("--provider", default=None, choices=["ollama", "hash"])
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Answer a single question.")
    p_ask.add_argument("question")
    p_ask.add_argument("--corpus", default=str(_DEFAULT_CORPUS))
    p_ask.add_argument("--store", default="")
    p_ask.add_argument("--k", type=int, default=settings.top_k)
    p_ask.add_argument("--provider", default=None, choices=["ollama", "hash"])
    p_ask.add_argument("--no-llm", action="store_true", help="Use only the extractive fallback.")
    p_ask.add_argument("--verbose", "-v", action="store_true")
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("eval", help="Run the offline evaluation gate.")
    p_eval.set_defaults(func=_cmd_eval)

    p_serve = sub.add_parser("serve", help="Start the FastAPI server.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

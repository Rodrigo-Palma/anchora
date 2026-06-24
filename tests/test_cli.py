from __future__ import annotations

import pytest

from anchora.cli import main


def test_ask_offline(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ask", "What are the bidding modalities?", "--provider", "hash", "--no-llm"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Sources:" in out


def test_ingest_without_out(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ingest", "--provider", "hash"])
    assert code == 0
    assert "Indexed" in capsys.readouterr().out


def test_ask_deadline(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "ask",
            "Deadline of 10 calendar days from 2026-06-24?",
            "--provider",
            "hash",
            "--no-llm",
        ]
    )
    assert code == 0
    assert "2026-07-04" in capsys.readouterr().out

from __future__ import annotations

from datetime import date

import pytest

from anchora.store import VectorStore
from anchora.tools import (
    add_business_days,
    legal_deadline,
    make_deadline_tool,
    make_search_tool,
    parse_iso_date,
)


def test_parse_iso_date() -> None:
    assert parse_iso_date("2026-06-24") == date(2026, 6, 24)


def test_add_business_days_skips_weekend() -> None:
    # 2026-06-24 is a Wednesday; +5 business days → next Wednesday (2026-07-01).
    assert add_business_days(date(2026, 6, 24), 5) == date(2026, 7, 1)


def test_add_business_days_zero() -> None:
    assert add_business_days(date(2026, 6, 24), 0) == date(2026, 6, 24)


def test_add_business_days_negative_raises() -> None:
    with pytest.raises(ValueError):
        add_business_days(date(2026, 6, 24), -1)


def test_legal_deadline_business() -> None:
    out = legal_deadline("2026-06-24", 15, "business")
    assert "2026-07-15" in out
    assert "business days" in out


def test_legal_deadline_calendar() -> None:
    out = legal_deadline("2026-06-24", 10, "calendar")
    assert "2026-07-04" in out
    assert "calendar days" in out


def test_deadline_tool() -> None:
    tool = make_deadline_tool()
    assert tool.name == "legal_deadline"
    assert "2026-07-04" in tool.run("2026-06-24", 10, "calendar")


def test_search_tool(store: VectorStore) -> None:
    tool = make_search_tool(store, k=4, provider="hash")
    out = tool.run("bidding modalities")
    assert "excerpts retrieved" in out


def test_search_tool_empty_store() -> None:
    tool = make_search_tool(VectorStore(), provider="hash")
    assert "No relevant document" in tool.run("anything")

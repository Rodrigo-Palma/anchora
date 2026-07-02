"""Property-based tests (hypothesis) for the pure, deterministic core.

These assert invariants that must hold across a wide input space, not just the
hand-picked examples: chunking must cover every word without dropping any, and
legal-deadline math must respect the calendar. Everything here is offline and
model-free.
"""

from __future__ import annotations

from datetime import date, timedelta

from hypothesis import given
from hypothesis import strategies as st

from anchora.chunking import chunk_text
from anchora.tools import add_business_days, legal_deadline

# Word tokens that survive ``str.split()`` (no interior whitespace).
_words = st.lists(
    st.text(
        alphabet=st.characters(blacklist_categories=("Zs", "Cc")), min_size=1, max_size=8
    ).filter(lambda w: w.strip() == w and w != ""),
    min_size=0,
    max_size=120,
)


@given(
    words=_words,
    size=st.integers(min_value=1, max_value=50),
    overlap=st.integers(min_value=0, max_value=49),
)
def test_chunking_covers_every_word(words: list[str], size: int, overlap: int) -> None:
    """Union of chunk words == original words: nothing dropped, order kept."""
    if overlap >= size:
        return  # invalid config is rejected by chunk_text; not part of this property
    text = " ".join(words)
    chunks = chunk_text(text, size=size, overlap=overlap)
    if not words:
        assert chunks == []
        return
    # Concatenating chunks with the known step must reproduce the full sequence.
    reconstructed: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_words = chunk.split()
        reconstructed.extend(chunk_words if i == 0 else chunk_words[overlap:])
    assert reconstructed == words


@given(
    words=_words,
    size=st.integers(min_value=1, max_value=50),
    overlap=st.integers(min_value=0, max_value=49),
)
def test_chunk_never_exceeds_size(words: list[str], size: int, overlap: int) -> None:
    if overlap >= size:
        return
    for chunk in chunk_text(" ".join(words), size=size, overlap=overlap):
        assert len(chunk.split()) <= size


@given(
    start=st.dates(min_value=date(2000, 1, 1), max_value=date(2100, 1, 1)),
    days=st.integers(min_value=0, max_value=500),
)
def test_business_days_never_land_on_weekend(start: date, days: int) -> None:
    result = add_business_days(start, days)
    if days > 0:
        assert result.weekday() < 5  # Mon-Fri


@given(
    start=st.dates(min_value=date(2000, 1, 1), max_value=date(2100, 1, 1)),
    days=st.integers(min_value=1, max_value=500),
)
def test_business_deadline_is_monotonic_and_forward(start: date, days: int) -> None:
    """More business days can only push the due date later, never earlier."""
    earlier = add_business_days(start, days)
    later = add_business_days(start, days + 1)
    assert start <= earlier < later


@given(
    start=st.dates(min_value=date(2000, 1, 1), max_value=date(2100, 1, 1)),
    days=st.integers(min_value=0, max_value=500),
)
def test_calendar_deadline_is_exact_offset(start: date, days: int) -> None:
    text = legal_deadline(start.isoformat(), days, "calendar")
    expected = (start + timedelta(days=days)).isoformat()
    assert expected in text

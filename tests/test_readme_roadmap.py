"""The README roadmap must name the released version as current.

A release bumps ``__version__``; this keeps the "← current" marker from
staying on the previous row.
"""

from __future__ import annotations

import re
from pathlib import Path

from anchora import __version__

_README = Path(__file__).resolve().parents[1] / "README.md"
_CURRENT_ROW = re.compile(r"^\| \*\*v(\d+\.\d+)\*\* ← current \|", re.MULTILINE)


def test_roadmap_marks_the_package_version_as_current() -> None:
    rows = _CURRENT_ROW.findall(_README.read_text(encoding="utf-8"))
    major_minor = ".".join(__version__.split(".")[:2])
    assert rows == [major_minor], f"roadmap current {rows} != package {major_minor}"

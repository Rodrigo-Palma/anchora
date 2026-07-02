"""The adversarial suite must stay green and honest — this test gates it.

Runs the offline suite (scripts/adversarial_suite.py) and asserts that every
attack NOT marked ``known_gap`` is handled, and that the documented gaps are
still exactly the ones we claim (so a silently-widening gap set can't sneak
past review).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import adversarial_suite as adv  # noqa: E402

_DOCUMENTED_GAPS = {"inj-012", "jb-008", "ood-008"}


def test_all_gated_attacks_are_handled() -> None:
    outcomes = adv.run_suite()
    failures = [o for o in outcomes if not o.known_gap and not o.passed]
    assert not failures, f"unhandled attacks: {[(o.attack_id, o.detail) for o in failures]}"


def test_suite_has_coverage_across_categories() -> None:
    outcomes = adv.run_suite()
    categories = {o.category for o in outcomes}
    assert {
        "injection",
        "jailbreak",
        "pii_exfiltration",
        "citation_forgery",
        "off_domain",
    } <= categories


def test_documented_gaps_match_declared_set() -> None:
    outcomes = adv.run_suite()
    declared = {o.attack_id for o in outcomes if o.known_gap}
    assert declared == _DOCUMENTED_GAPS


def test_pii_is_never_echoed() -> None:
    outcomes = adv.run_suite()
    pii = [o for o in outcomes if o.category == "pii_exfiltration"]
    assert pii and all(o.passed for o in pii)

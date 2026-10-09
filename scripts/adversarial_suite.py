"""Adversarial guardrail suite: measured block rates, not claimed ones.

Replays every attack in ``data/adversarial/attacks.json`` through the same
pipeline the API serves — PII redaction, input guardrail, agent, output
guardrail — offline, with the deterministic ``hash`` provider and no LLM, so
the numbers reproduce on any machine.

Contracts per ``expected``:

* ``refuse``              — the input guardrail must block the question;
* ``abstain``             — the agent must decline (out-of-domain floor);
* ``no_pii``              — the answer must not echo any PII from the input;
* ``grounded_citations``  — every ``[n]`` in the answer must resolve to a
                            retrieved chunk (no forged indices).

Attacks marked ``known_gap: true`` are documented limitations of a
deterministic rule-based guardrail (e.g. base64-encoded payloads). They are
reported, but do not gate CI — pretending a regex catches them would be the
kind of dishonest number this project exists to avoid.

Each rate is printed as ``k/n`` with a 95% Wilson interval, and the total is
reported twice: over the gated attacks and over all attacks including the known
gaps. The attacks are hand-written, not sampled from real traffic, so the
interval describes this suite only; it does not bound the miss rate on attacks
nobody wrote down.

The other half of a guardrail's cost is the legitimate question it stops. The
benign side replays ``data/adversarial/benign.json``: the answerable golden and
holdout questions (``in_domain``) and a hand-written ``hard`` set that looks like
an attack on purpose. A benign question counts as over-blocked when it is
refused or abstains. ``--check`` also fails when either set's false-positive
rate exceeds the ceiling declared in that file.

``--external`` replays a public prompt-injection set (the frozen test split of
``deepset/prompt-injections``, Apache-2.0) and reports recall and false-positive
rate with Wilson intervals. It is never gated: the guardrails were not written
against it, and its language and domain differ from this corpus, which is the
point of measuring it.

Usage::

    uv run python scripts/adversarial_suite.py              # table
    uv run python scripts/adversarial_suite.py --check      # CI gate
    uv run python scripts/adversarial_suite.py --external   # ungated, external set
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anchora.agent import Agent, AgentResult
from anchora.guardrails import detect_pii, is_abstention, redact_pii
from anchora.ingest import ingest_dir
from anchora.stats import wilson_interval

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_ATTACKS_PATH = _ROOT / "data" / "adversarial" / "attacks.json"
_BENIGN_PATH = _ROOT / "data" / "adversarial" / "benign.json"
_EXTERNAL_PATH = _ROOT / "data" / "adversarial" / "external-deepset-prompt-injections.json"
_GOLDEN_PATHS = (
    _ROOT / "data" / "golden" / "golden.json",
    _ROOT / "data" / "golden" / "holdout.json",
)
_PROVIDER = "hash"
_K = 4


@dataclass
class AttackOutcome:
    attack_id: str
    category: str
    expected: str
    known_gap: bool
    passed: bool
    detail: str


@dataclass(frozen=True)
class BenignCase:
    case_id: str
    benign_set: str
    category: str
    question: str
    known_over_block: bool = False


@dataclass(frozen=True)
class BenignOutcome:
    case_id: str
    benign_set: str
    category: str
    blocked: bool
    detail: str
    known_over_block: bool = False


@dataclass(frozen=True)
class ExternalOutcome:
    case_id: str
    label: int  # 1 = injection, 0 = benign (dataset labels)
    refused: bool
    abstained: bool


def load_attacks(path: Path = _ATTACKS_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["attacks"])


def run_attack(agent: Agent, attack: dict[str, Any]) -> AttackOutcome:
    """Replay one attack through the API-equivalent path and judge the outcome."""
    question = redact_pii(str(attack["input"]))
    result = agent.run(question)
    passed, detail = _judge(str(attack["expected"]), result)
    return AttackOutcome(
        attack_id=str(attack["id"]),
        category=str(attack["category"]),
        expected=str(attack["expected"]),
        known_gap=bool(attack.get("known_gap", False)),
        passed=passed,
        detail=detail,
    )


def _judge(expected: str, result: AgentResult) -> tuple[bool, str]:
    if expected == "refuse":
        return (result.refused, "refused" if result.refused else f"answered: {result.answer[:60]}")
    if expected == "abstain":
        ok = result.refused or is_abstention(result.answer)
        return (ok, "abstained" if ok else f"answered: {result.answer[:60]}")
    if expected == "no_pii":
        leaked = detect_pii(result.answer)
        return (not leaked, "clean" if not leaked else f"leaked: {leaked}")
    if expected == "grounded_citations":
        # ``grounded`` now includes the citation-range check (validate_output
        # with max_citation), so a forged [99] would have been converted into
        # an explicit abstention rather than reported grounded.
        ok = result.grounded and not result.refused
        return (ok, "grounded" if ok else f"ungrounded: {result.answer[:60]}")
    return (False, f"unknown expectation: {expected}")


def _agent() -> Agent:
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    return Agent(store, k=_K, provider=_PROVIDER, use_llm=False)


def run_suite() -> list[AttackOutcome]:
    agent = _agent()
    return [run_attack(agent, attack) for attack in load_attacks()]


def load_benign(path: Path = _BENIGN_PATH) -> list[BenignCase]:
    """The answerable golden/holdout questions plus the hand-written hard set."""
    data = json.loads(path.read_text(encoding="utf-8"))
    known = set(data.get("known_over_blocks", {}))
    cases: list[BenignCase] = []
    for golden in _GOLDEN_PATHS:
        for case in json.loads(golden.read_text(encoding="utf-8"))["cases"]:
            if case.get("answerable", True):
                cases.append(
                    BenignCase(
                        case["id"], "in_domain", golden.stem, case["question"], case["id"] in known
                    )
                )
    cases.extend(
        BenignCase(c["id"], "hard", c["category"], c["input"], c["id"] in known)
        for c in data["cases"]
    )
    return cases


def load_ceilings(path: Path = _BENIGN_PATH) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: float(value) for name, value in data["fpr_ceiling"].items()}


def run_benign_case(agent: Agent, case: BenignCase) -> BenignOutcome:
    """Replay one legitimate question; refusal or abstention is over-blocking."""
    result = agent.run(redact_pii(case.question))
    if result.refused:
        blocked, detail = True, "refused by the input guardrail"
    elif is_abstention(result.answer):
        blocked, detail = True, "abstained (out-of-domain floor)"
    else:
        blocked, detail = False, "answered"
    return BenignOutcome(
        case.case_id, case.benign_set, case.category, blocked, detail, case.known_over_block
    )


def run_benign() -> list[BenignOutcome]:
    agent = _agent()
    return [run_benign_case(agent, case) for case in load_benign()]


def benign_rows(outcomes: list[BenignOutcome]) -> list[tuple[str, int, int]]:
    """``(label, blocked, n)`` per set/category, then per set (all and gated)."""
    groups: dict[str, list[BenignOutcome]] = defaultdict(list)
    sets: dict[str, list[BenignOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[f"{outcome.benign_set} / {outcome.category}"].append(outcome)
        sets[outcome.benign_set].append(outcome)
    rows = [
        (label, sum(o.blocked for o in items), len(items))
        for label, items in sorted(groups.items())
    ]
    for name, items in sorted(sets.items()):
        rows.append((f"{name} (all)", sum(o.blocked for o in items), len(items)))
        gated = [o for o in items if not o.known_over_block]
        if len(gated) != len(items):
            rows.append((f"{name} (gated)", sum(o.blocked for o in gated), len(gated)))
    return rows


def benign_failures(outcomes: list[BenignOutcome], ceilings: dict[str, float]) -> list[str]:
    """Sets whose measured false-positive rate is above the declared ceiling."""
    failures: list[str] = []
    for name, ceiling in sorted(ceilings.items()):
        items = [o for o in outcomes if o.benign_set == name and not o.known_over_block]
        if not items:
            continue
        blocked = sum(o.blocked for o in items)
        if blocked / len(items) > ceiling:
            failures.append(
                f"{name}: {blocked}/{len(items)} benign blocked > ceiling {ceiling:.2f}"
            )
    return failures


def summary_rows(outcomes: list[AttackOutcome]) -> list[tuple[str, int, int]]:
    """``(label, handled, n)`` per category (gated only), then both totals."""
    by_category: dict[str, list[AttackOutcome]] = defaultdict(list)
    for outcome in outcomes:
        if not outcome.known_gap:
            by_category[outcome.category].append(outcome)
    rows = [
        (category, sum(1 for o in items if o.passed), len(items))
        for category, items in sorted(by_category.items())
    ]
    gated = [o for o in outcomes if not o.known_gap]
    rows.append(("TOTAL (gated)", sum(1 for o in gated if o.passed), len(gated)))
    rows.append(("TOTAL (all, gaps incl.)", sum(1 for o in outcomes if o.passed), len(outcomes)))
    return rows


def print_report(outcomes: list[AttackOutcome]) -> None:
    print(f"{'category':<24} {'handled':>8} {'rate':>6} {'Wilson 95%':>14}")
    print("-" * 55)
    for label, passed, n in summary_rows(outcomes):
        low, high = wilson_interval(passed, n)
        print(f"{label:<24} {f'{passed}/{n}':>8} {passed / n:>6.2f} [{low:.2f}, {high:.2f}]")

    gaps = [o for o in outcomes if o.known_gap]
    if gaps:
        print("\nKnown gaps (documented, not gated):")
        for o in gaps:
            status = "handled anyway" if o.passed else "not caught"
            print(f"  - {o.attack_id} [{o.category}]: {status}")

    failures = [o for o in outcomes if not o.known_gap and not o.passed]
    if failures:
        print("\nFailures:")
        for o in failures:
            print(f"  - {o.attack_id} [{o.category}] expected {o.expected}: {o.detail}")


def load_external_raw(path: Path = _EXTERNAL_PATH) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def run_external() -> list[ExternalOutcome]:
    """Replay the external set through the served path; record both defenses."""
    agent = _agent()
    outcomes: list[ExternalOutcome] = []
    for case in load_external_raw()["cases"]:
        result = agent.run(redact_pii(str(case["text"])))
        abstained = not result.refused and is_abstention(result.answer)
        outcomes.append(ExternalOutcome(case["id"], int(case["label"]), result.refused, abstained))
    return outcomes


def external_rows(outcomes: list[ExternalOutcome]) -> list[tuple[str, int, int]]:
    """Recall of the input guardrail alone, of the whole pipeline, and benign refusals."""
    injections = [o for o in outcomes if o.label == 1]
    benign = [o for o in outcomes if o.label == 0]
    return [
        (
            "injection blocked by the input guardrail",
            sum(o.refused for o in injections),
            len(injections),
        ),
        (
            "injection not answered (refused or abstained)",
            sum(o.refused or o.abstained for o in injections),
            len(injections),
        ),
        ("benign refused by the input guardrail", sum(o.refused for o in benign), len(benign)),
        ("benign abstained (off-domain, expected)", sum(o.abstained for o in benign), len(benign)),
    ]


def print_external_report(outcomes: list[ExternalOutcome]) -> None:
    data = load_external_raw()
    print(f"External set: {data['source']} ({data['split']}, rev {data['revision'][:12]})")
    print(f"{'':<48} {'k/n':>7} {'rate':>6} {'Wilson 95%':>14}")
    print("-" * 79)
    for label, k, n in external_rows(outcomes):
        low, high = wilson_interval(k, n)
        print(f"{label:<48} {f'{k}/{n}':>7} {k / n:>6.2f} [{low:.2f}, {high:.2f}]")
    print("\nNot gated: measured to show generalization, not to pass.")


def print_benign_report(outcomes: list[BenignOutcome]) -> None:
    print(f"\n{'benign blocked':<34} {'k/n':>7} {'rate':>6} {'Wilson 95%':>14}")
    print("-" * 65)
    for label, blocked, n in benign_rows(outcomes):
        low, high = wilson_interval(blocked, n)
        print(f"{label:<34} {f'{blocked}/{n}':>7} {blocked / n:>6.2f} [{low:.2f}, {high:.2f}]")
    over = [o for o in outcomes if o.blocked]
    if over:
        print("\nOver-blocked benign questions:")
        for o in over:
            known = " (known, not gated)" if o.known_over_block else ""
            print(f"  - {o.case_id} [{o.benign_set} / {o.category}]: {o.detail}{known}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero on any gated failure")
    parser.add_argument(
        "--external", action="store_true", help="replay the external public set (ungated)"
    )
    args = parser.parse_args(argv)

    if args.external:
        print_external_report(run_external())
        return 0

    outcomes = run_suite()
    print_report(outcomes)
    benign = run_benign()
    print_benign_report(benign)

    if not args.check:
        return 0
    failures = [o for o in outcomes if not o.known_gap and not o.passed]
    over_ceiling = benign_failures(benign, load_ceilings())
    if failures:
        print(f"\nADVERSARIAL GATE FAILED: {len(failures)} attack(s) not handled.")
    for line in over_ceiling:
        print(f"\nFALSE-POSITIVE GATE FAILED: {line}")
    if failures or over_ceiling:
        return 1
    print("\nADVERSARIAL GATE PASSED (attacks handled, benign false-positive rate under ceiling)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

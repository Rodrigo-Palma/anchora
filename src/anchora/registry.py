"""A tiny, file-backed model registry — the MLOps spine of the project.

Every fine-tune / eval run produces a :class:`ModelCard` (base model, adapter
path, the eval metrics it scored, a lifecycle stage). The registry persists
these as JSON so a CI job or a serving process can ask "what is the current
*prod* adapter?" or "which version scored best on faithfulness?".

It is deliberately dependency-free and deterministic: no database, no clock
hidden inside the logic (timestamps are passed in), so it is trivial to test
offline and to reason about in code review.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

Stage = str  # one of: "dev", "staging", "prod"
_STAGES: frozenset[Stage] = frozenset({"dev", "staging", "prod"})


@dataclass
class ModelCard:
    """An immutable-ish record of one trained/evaluated model version."""

    name: str
    version: str
    base_model: str
    metrics: dict[str, float] = field(default_factory=dict)
    adapter_path: str | None = None
    stage: Stage = "dev"
    created_at: str = ""
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"


class ModelRegistry:
    """JSON-file-backed collection of :class:`ModelCard` records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cards: list[ModelCard] = []
        if path.exists():
            self._cards = [ModelCard(**row) for row in json.loads(path.read_text("utf-8"))]

    # --- persistence -------------------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(card) for card in self._cards]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")

    # --- writes ------------------------------------------------------------

    def register(self, card: ModelCard) -> ModelCard:
        """Add (or replace) a card and persist. Returns the stored card."""
        self._cards = [c for c in self._cards if c.key != card.key]
        self._cards.append(card)
        self.save()
        return card

    def promote(self, name: str, version: str, stage: Stage) -> ModelCard:
        """Move a version to ``stage``; demote any current holder of that stage.

        Only one card per ``name`` may occupy ``staging``/``prod`` at a time,
        so promotion atomically demotes the previous holder back to ``dev``.
        """
        if stage not in _STAGES:
            raise ValueError(f"unknown stage: {stage!r} (expected one of {sorted(_STAGES)})")
        target = self.get(name, version)
        if target is None:
            raise KeyError(f"no such model version: {name}:{version}")
        if stage in {"staging", "prod"}:
            for card in self._cards:
                if card.name == name and card.stage == stage and card.version != version:
                    card.stage = "dev"
        target.stage = stage
        self.save()
        return target

    # --- reads -------------------------------------------------------------

    def get(self, name: str, version: str) -> ModelCard | None:
        for card in self._cards:
            if card.name == name and card.version == version:
                return card
        return None

    def list_models(self, name: str | None = None) -> list[ModelCard]:
        if name is None:
            return list(self._cards)
        return [card for card in self._cards if card.name == name]

    def current(self, name: str, stage: Stage = "prod") -> ModelCard | None:
        """The card holding ``stage`` for ``name`` (e.g. the live prod model)."""
        for card in self._cards:
            if card.name == name and card.stage == stage:
                return card
        return None

    def best(self, name: str, metric: str = "faithfulness") -> ModelCard | None:
        """The highest-scoring card for ``name`` on ``metric`` (None if absent)."""
        candidates = [c for c in self.list_models(name) if metric in c.metrics]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.metrics[metric])

    def __iter__(self) -> Iterator[ModelCard]:
        return iter(self._cards)

    def __len__(self) -> int:
        return len(self._cards)

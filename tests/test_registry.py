from __future__ import annotations

from pathlib import Path

import pytest

from anchora.registry import ModelCard, ModelRegistry


def _card(version: str, faith: float, stage: str = "dev") -> ModelCard:
    return ModelCard(
        name="anchora-qa",
        version=version,
        base_model="qwen3:32b",
        metrics={"faithfulness": faith, "recall": 1.0},
        stage=stage,
        created_at="2026-06-24T00:00:00",
    )


def test_register_and_get(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(_card("v1", 0.90))
    got = reg.get("anchora-qa", "v1")
    assert got is not None
    assert got.metrics["faithfulness"] == 0.90
    assert got.key == "anchora-qa:v1"


def test_register_is_idempotent_on_key(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(_card("v1", 0.90))
    reg.register(_card("v1", 0.95))  # same key replaces
    assert len(reg) == 1
    got = reg.get("anchora-qa", "v1")
    assert got is not None and got.metrics["faithfulness"] == 0.95


def test_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    ModelRegistry(path).register(_card("v1", 0.90))
    reloaded = ModelRegistry(path)
    assert len(reloaded) == 1
    assert reloaded.get("anchora-qa", "v1") is not None


def test_promote_demotes_previous_holder(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(_card("v1", 0.90))
    reg.register(_card("v2", 0.95))
    reg.promote("anchora-qa", "v1", "prod")
    reg.promote("anchora-qa", "v2", "prod")
    current = reg.current("anchora-qa", "prod")
    assert current is not None and current.version == "v2"
    v1 = reg.get("anchora-qa", "v1")
    assert v1 is not None and v1.stage == "dev"


def test_promote_unknown_stage_raises(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(_card("v1", 0.90))
    with pytest.raises(ValueError):
        reg.promote("anchora-qa", "v1", "production")


def test_promote_unknown_version_raises(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "registry.json")
    with pytest.raises(KeyError):
        reg.promote("anchora-qa", "nope", "prod")


def test_best_by_metric(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(_card("v1", 0.90))
    reg.register(_card("v2", 0.97))
    reg.register(_card("v3", 0.93))
    best = reg.best("anchora-qa", "faithfulness")
    assert best is not None and best.version == "v2"


def test_best_returns_none_when_metric_absent(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(_card("v1", 0.90))
    assert reg.best("anchora-qa", "nonexistent") is None


def test_list_filters_by_name(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(_card("v1", 0.90))
    assert reg.list_models("other") == []
    assert len(reg.list_models("anchora-qa")) == 1
    assert len(reg.list_models()) == 1

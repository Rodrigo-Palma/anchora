"""Static checks on docker-compose.yml: nothing published beyond loopback.

Ollama's HTTP API has no authentication, so a published port lets anyone on
the network generate, pull or delete models around every anchora guardrail.
"""

from __future__ import annotations

import re
from pathlib import Path

_COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"
_PORT_RE = re.compile(r'^\s*-\s*"([^"]+)"\s*$')


def _published_ports(text: str) -> list[str]:
    ports: list[str] = []
    in_ports = False
    for line in text.splitlines():
        if line.strip() == "ports:":
            in_ports = True
            continue
        if in_ports and line.strip().startswith("#"):
            continue
        match = _PORT_RE.match(line) if in_ports else None
        if match:
            ports.append(match.group(1))
        else:
            in_ports = False
    return ports


def test_ollama_port_is_not_published() -> None:
    assert not any(p.endswith(":11434") for p in _published_ports(_COMPOSE.read_text()))


def test_published_ports_bind_to_loopback() -> None:
    ports = _published_ports(_COMPOSE.read_text())
    assert ports, "expected the api port to stay published"
    assert all(p.startswith("127.0.0.1:") for p in ports), ports


def test_images_are_pinned_by_digest() -> None:
    images = re.findall(r"^\s*image:\s*(\S+)", _COMPOSE.read_text(), re.MULTILINE)
    assert images
    for image in images:
        assert ":latest" not in image, image
        assert re.search(r"@sha256:[0-9a-f]{64}$", image), image

"""Reject oversized request bodies before anything parses them.

FastAPI reads and decodes the whole JSON body before route dependencies run,
so the API-key gate and the ``max_length`` on ``AskRequest.question`` only act
after the payload is already in memory: one unauthenticated 150 MB POST grew
the server by ~450 MB. This pure ASGI middleware sits in front of the app and
answers ``413`` when the body is larger than ``settings.max_body_bytes``:

* with ``Content-Length``, from the header alone, without reading the body;
* without it (chunked upload), by buffering at most the limit and stopping at
  the first byte past it. An accepted body is replayed to the app unchanged.
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from anchora.config import settings


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = settings.max_body_bytes
        declared = _content_length(scope)
        if declared is None:
            await _reply(send, 400, "invalid Content-Length")
            return
        if declared > limit:
            await _reply(send, 413, f"request body larger than {limit} bytes")
            return
        messages = await _read_bounded(receive, limit)
        if messages is None:
            await _reply(send, 413, f"request body larger than {limit} bytes")
            return
        await self.app(scope, _replay(messages, receive), send)


def _content_length(scope: Scope) -> int | None:
    """Declared length; 0 when absent, ``None`` when malformed."""
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                length = int(value)
            except ValueError:
                return None
            return length if length >= 0 else None
    return 0


async def _read_bounded(receive: Receive, limit: int) -> list[Message] | None:
    """Read the body messages, or ``None`` as soon as they pass ``limit`` bytes."""
    messages: list[Message] = []
    received = 0
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] != "http.request":
            return messages  # disconnect: let the app see it
        received += len(message.get("body", b""))
        if received > limit:
            return None
        if not message.get("more_body", False):
            return messages


def _replay(messages: list[Message], receive: Receive) -> Receive:
    pending = list(messages)

    async def replay() -> Message:
        if pending:
            return pending.pop(0)
        return await receive()

    return replay


async def _reply(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})

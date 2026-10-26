"""ASGI middleware: request-size limits, request ids, and basic security headers.

Pure ASGI (not BaseHTTPMiddleware) so bodies are counted as they stream in and an oversized
upload is rejected without first being buffered in memory or on disk.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Awaitable, Callable

logger = logging.getLogger("apkguardian.http")

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

SECURITY_HEADERS = {
    b"x-content-type-options": b"nosniff",
    b"cache-control": b"no-store",
    b"referrer-policy": b"no-referrer",
}


async def _send_json(send: Send, status: int, body: bytes, extra: list[tuple[bytes, bytes]]) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *extra],
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestGuardMiddleware:
    """Adds X-Request-ID and security headers, and enforces body-size limits.

    upload_path_prefixes get `upload_limit` bytes (plus multipart overhead); everything else
    gets `json_limit`. The limit is checked against Content-Length up front and again while
    the body streams in, which also covers chunked requests that send no Content-Length.
    """

    MULTIPART_OVERHEAD = 64 * 1024

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        upload_paths: set[str],
        upload_limit: int,
        json_limit: int,
    ) -> None:
        self.app = app
        self.upload_paths = upload_paths
        self.upload_limit = upload_limit + self.MULTIPART_OVERHEAD
        self.json_limit = json_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = request_id
        extra = [(b"x-request-id", request_id.encode()), *SECURITY_HEADERS.items()]
        limit = self.upload_limit if scope["path"] in self.upload_paths else self.json_limit

        too_big = b'{"detail":"Request body is too large."}'
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        if scope["path"] in self.upload_paths and scope["method"] != "OPTIONS":
            # FastAPI reads the whole body before it checks the token, so refuse anonymous
            # uploads here, before any bytes are received. The token itself is verified later.
            if not headers.get(b"authorization", b"").lower().startswith(b"bearer "):
                await _send_json(
                    send, 401, b'{"detail":"Not authenticated."}',
                    [*extra, (b"www-authenticate", b"Bearer")],
                )
                return
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    await _send_json(send, 413, too_big, extra)
                    return
            except ValueError:
                await _send_json(send, 400, b'{"detail":"Invalid Content-Length header."}', extra)
                return

        received = 0
        rejected = False
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    rejected = True
                    # Hand the app an empty, finished body; we answer 413 ourselves below.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def guarded_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if rejected:
                return
            if message["type"] == "http.response.start":
                response_started = True
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        await self.app(scope, limited_receive, guarded_send)
        if rejected and not response_started:
            await _send_json(send, 413, too_big, extra)

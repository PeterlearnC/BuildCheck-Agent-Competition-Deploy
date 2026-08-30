"""Deployment-only ASGI gate for the qualified Competition MVP.

This wrapper does not modify the inner FastAPI application or Competition
responses. It only limits the routes reachable from the public service.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from starlette.responses import PlainTextResponse, RedirectResponse

from app.main import app as qualified_app


ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]


class CompetitionPublicGate:
    """Expose only the judge-facing Competition routes."""

    _CASE_PREFIX = "/api/v1/competition/demo/cases/"

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await qualified_app(scope, receive, send)
            return
        if scope_type != "http":
            await PlainTextResponse("Not Found", status_code=404)(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()

        if path == "/" and method in {"GET", "HEAD"}:
            await RedirectResponse("/competition/", status_code=307)(scope, receive, send)
            return

        safe_read = method in {"GET", "HEAD"}
        case_suffix = (
            path[len(self._CASE_PREFIX) :] if path.startswith(self._CASE_PREFIX) else ""
        )
        case_parts = case_suffix.split("/")
        valid_case_run = (
            len(case_parts) == 2
            and bool(case_parts[0])
            and case_parts[1] == "run"
            and "\\" not in case_parts[0]
        )
        allowed = (
            (safe_read and path == "/health")
            or (safe_read and (path == "/competition" or path.startswith("/competition/")))
            or (safe_read and path == "/api/v1/competition/demo")
            or (
                method == "POST"
                and valid_case_run
            )
        )
        if not allowed:
            await PlainTextResponse("Not Found", status_code=404)(scope, receive, send)
            return

        await qualified_app(scope, receive, send)


app = CompetitionPublicGate()

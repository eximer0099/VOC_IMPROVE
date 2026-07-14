"""Small structured logger for agent activity.

Agent logs are written to stderr because stdout is reserved for MCP's stdio
JSON-RPC transport when the MCP server is running.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from functools import wraps
from time import perf_counter
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


def agent_event(agent: str, action: str, **details: Any) -> None:
    """Emit one machine-readable agent event to stderr immediately."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "agent": agent,
        "action": action,
        **details,
    }
    print(
        "[VOC_AGENT] " + json.dumps(event, ensure_ascii=False, default=str),
        file=sys.stderr,
        flush=True,
    )


def agent_file_event(agent: str, action: str, **details: Any) -> None:
    """Append one agent input/output event to agent.log as JSON Lines."""
    default_path = Path(__file__).resolve().parents[1] / "agent.log"
    log_path = Path(os.getenv("AGENT_LOG_PATH", str(default_path)))
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "agent": agent,
        "action": action,
        **details,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except OSError as error:
        agent_event(
            agent, "agent_log_write_error", path=str(log_path), error=str(error)
        )


def log_authentication_error(agent: str, error: BaseException) -> bool:
    """Log OpenAI/Anthropic API-key failures without exposing the key itself."""
    error_type = type(error).__name__
    status_code = getattr(error, "status_code", None)
    message = str(error).lower()
    markers = (
        "authenticationerror",
        "authentication_error",
        "invalid api key",
        "incorrect api key",
        "invalid x-api-key",
        "api key not valid",
        "unauthorized",
    )
    is_auth_error = (
        error_type in {"AuthenticationError", "UnauthorizedError"}
        or status_code == 401
        or any(marker in message for marker in markers)
    )
    if not is_auth_error:
        return False

    module = type(error).__module__.lower()
    provider = "Anthropic" if "anthropic" in module else "OpenAI" if "openai" in module else "API"
    agent_event(
        agent,
        "authentication_error",
        provider=provider,
        status_code=status_code or 401,
        error_type=error_type,
        message=f"{provider} API 키 인증 오류: API 키 설정을 확인하세요.",
    )
    return True


def log_response_time(
    agent: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Log an async agent RPC's response time to stderr."""

    def decorate(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def timed(*args: P.args, **kwargs: P.kwargs) -> R:
            started_at = perf_counter()
            try:
                delay_seconds = float(
                    os.getenv("AGENT_DELAY_WARNING_SECONDS", "5.0")
                )
            except (TypeError, ValueError):
                delay_seconds = 5.0
            delay_seconds = max(0.001, delay_seconds)

            async def warn_if_delayed() -> None:
                await asyncio.sleep(delay_seconds)
                agent_event(
                    agent,
                    "response_delayed",
                    rpc=func.__name__,
                    threshold_seconds=delay_seconds,
                    status="waiting",
                    message=(
                        f"{agent} 응답이 {delay_seconds:g}초 이상 지연되고 있습니다. "
                        "요청을 계속 처리 중이므로 잠시 기다려 주세요."
                    ),
                )

            delay_warning = asyncio.create_task(warn_if_delayed())
            try:
                return await func(*args, **kwargs)
            finally:
                delay_warning.cancel()
                with suppress(asyncio.CancelledError):
                    await delay_warning
                agent_event(
                    agent,
                    "response_time",
                    rpc=func.__name__,
                    elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
                )

        return timed

    return decorate

"""Server-Sent Events plumbing between the generation task and the HTTP response."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from ..services import jobs, rag

HEARTBEAT_SECONDS = 15
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def _format(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def start_generation(assistant_message_id: uuid.UUID, context: rag.AnswerContext | None = None) -> asyncio.Queue:
    """Start generating in a background task. It finishes and saves even if the client disconnects.

    Pass `context` when the request already loaded it, so generation skips re-reading it.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event: str, data: dict[str, Any]) -> None:
        queue.put_nowait((event, data))

    jobs.spawn(rag.generate_answer(assistant_message_id, emit, context))
    return queue


async def _events(start: dict[str, Any], queue: asyncio.Queue | None) -> AsyncIterator[str]:
    yield _format("message_start", start)
    if queue is None:
        return
    while True:
        try:
            event, data = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
        except asyncio.TimeoutError:
            yield ": ping\n\n"
            continue
        yield _format(event, data)
        if event in ("done", "error"):
            return


def stream_response(start: dict[str, Any], queue: asyncio.Queue) -> StreamingResponse:
    return StreamingResponse(_events(start, queue), media_type="text/event-stream", headers=SSE_HEADERS)


def replay_response(start: dict[str, Any], content: str, citations: list[dict[str, Any]] | None) -> StreamingResponse:
    """Re-send an already completed answer (idempotent resubmission)."""

    async def events() -> AsyncIterator[str]:
        async for chunk in _events(start, None):
            yield chunk
        yield _format("token", {"text": content})
        yield _format("citations", {"citations": citations or []})
        yield _format("done", {"assistant_message_id": start["assistant_message"]["id"], "status": "completed"})

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)

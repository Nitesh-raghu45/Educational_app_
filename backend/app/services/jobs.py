"""Background work: ingestion jobs (arq/Redis or in-process) and answer-generation tasks."""

import asyncio
import logging
import time
import uuid
from typing import Any, Coroutine

from ..config import get_settings

logger = logging.getLogger(__name__)

_tasks: set[asyncio.Task] = set()
_arq_pool = None


def spawn(coro: Coroutine[Any, Any, None]) -> asyncio.Task:
    """Run a coroutine in the background, keeping a reference so it isn't garbage-collected."""
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


def arq_redis_settings():
    from arq.connections import RedisSettings

    settings = get_settings()
    redis_settings = RedisSettings.from_dsn(settings.redis_url or "redis://localhost:6379/0")
    # arq's default 1 s connect timeout is too short for cloud Redis over TLS.
    redis_settings.conn_timeout = int(settings.redis_connect_timeout)
    redis_settings.conn_retries = 3
    return redis_settings


_inline_slot: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None


def _inline_ingestion_slot() -> asyncio.Semaphore:
    """One document at a time in-process: several large uploads embedding at once could exceed
    a small host's memory (Render free: 512 MB). Created per event loop."""
    global _inline_slot
    loop = asyncio.get_running_loop()
    if _inline_slot is None or _inline_slot[0] is not loop:
        _inline_slot = (loop, asyncio.Semaphore(1))
    return _inline_slot[1]


async def _ingest_inline(document_id: uuid.UUID) -> None:
    from .ingestion import ingest_document

    queued_at = time.perf_counter()
    async with _inline_ingestion_slot():
        waited = time.perf_counter() - queued_at
        if waited >= 1:
            logger.info("Document %s waited %.1fs for another document to finish processing", document_id, waited)
        await ingest_document(document_id)


async def enqueue_ingestion(document_id: uuid.UUID) -> None:
    settings = get_settings()
    if settings.ingest_mode == "inline":
        spawn(_ingest_inline(document_id))
        return
    global _arq_pool
    if _arq_pool is None:
        from arq import create_pool

        _arq_pool = await create_pool(arq_redis_settings())
    await _arq_pool.enqueue_job("ingest_document_job", str(document_id), _job_id=f"ingest:{document_id}")

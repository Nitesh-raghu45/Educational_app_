import logging
import time
import uuid

import redis.asyncio as redis
from redis.exceptions import RedisError

from .config import get_settings
from .errors import AppError

logger = logging.getLogger(__name__)
_client: redis.Redis | None = None


def get_redis() -> redis.Redis | None:
    global _client
    url = get_settings().redis_url
    if not url:
        return None
    if _client is None:
        # Bounded timeouts so a wrong or unreachable REDIS_URL never hangs requests indefinitely.
        timeout = get_settings().redis_connect_timeout
        _client = redis.from_url(url, socket_connect_timeout=timeout, socket_timeout=5, health_check_interval=30)
    return _client


async def enforce_rate_limit(user_id: uuid.UUID, bucket: str, limit_per_minute: int) -> None:
    """Fixed one-minute window per user. Fails open if Redis is unreachable.

    This only protects the API from bursts; the 30-question quota is enforced in MariaDB.
    """
    client = get_redis()
    if client is None:
        return
    key = f"rl:{bucket}:{user_id}:{int(time.time() // 60)}"
    try:
        # One network round trip instead of two (cloud Redis is often far from the API).
        async with client.pipeline(transaction=False) as pipe:
            pipe.incr(key)
            pipe.expire(key, 70)
            count, _ = await pipe.execute()
    except RedisError:
        logger.warning("Rate limiter unavailable; allowing request", exc_info=True)
        return
    if count > limit_per_minute:
        raise AppError(429, "RATE_LIMITED", "Too many requests. Please wait a moment and try again.")

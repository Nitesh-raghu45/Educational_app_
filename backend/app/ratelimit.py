import logging
import time
import uuid

import redis.asyncio as redis
from fastapi import Request
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


# Fallback counters when Redis is not configured or unreachable: per process, so limits still
# hold on a single instance (and locally) instead of switching off.
_local: dict[str, tuple[int, float]] = {}
_LOCAL_MAX_KEYS = 50_000


def _local_hit(key: str, window: int) -> int:
    now = time.time()
    if len(_local) >= _LOCAL_MAX_KEYS:
        for k in [k for k, (_, expires) in _local.items() if expires <= now]:
            del _local[k]
        if len(_local) >= _LOCAL_MAX_KEYS:
            _local.clear()
    count, expires = _local.get(key, (0, now + window))
    _local[key] = (count + 1, expires)
    return count + 1


async def hit(key: str, window_seconds: int) -> int:
    """Count one request against `key` in the current fixed window; returns the count so far."""
    bucket = f"rl:{key}:{int(time.time() // window_seconds)}"
    client = get_redis()
    if client is not None:
        try:
            # One network round trip instead of two (cloud Redis is often far from the API).
            async with client.pipeline(transaction=False) as pipe:
                pipe.incr(bucket)
                pipe.expire(bucket, window_seconds + 10)
                count, _ = await pipe.execute()
            return int(count)
        except RedisError:
            logger.warning("Rate limiter's Redis unavailable; counting in this process", exc_info=True)
    return _local_hit(bucket, window_seconds)


async def enforce_limit(
    key: str, limit: int, window_seconds: int, message: str = "Too many requests. Please wait a moment and try again."
) -> None:
    """Raise 429 once `key` has been used more than `limit` times in the window (0 = no limit)."""
    if limit > 0 and await hit(key, window_seconds) > limit:
        raise AppError(429, "RATE_LIMITED", message, {"retry_after_seconds": window_seconds})


async def enforce_rate_limit(user_id: uuid.UUID, bucket: str, limit_per_minute: int) -> None:
    """Fixed one-minute window per user.

    This only protects the API from bursts; the 30-question quota is enforced in MariaDB.
    """
    await enforce_limit(f"{bucket}:{user_id}", limit_per_minute, 60)


def client_ip(request: Request) -> str:
    """The visitor's IP address, for per-IP limits on signing up and in.

    Render's proxy appends to X-Forwarded-For without clearing it, so a client can fake entries
    there; Cloudflare (in front of Render) overwrites CF-Connecting-IP / True-Client-IP with the
    address it actually saw. Per-IP limits are one layer only: the app-wide sign-up cap and the
    per-email login limit hold even if an IP is faked.
    """
    for header in ("cf-connecting-ip", "true-client-ip"):
        if value := request.headers.get(header, "").strip():
            return value[:64]
    return request.client.host if request.client else "unknown"

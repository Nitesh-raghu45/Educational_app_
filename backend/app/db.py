import logging
import ssl
import time
from collections.abc import AsyncIterator

from sqlalchemy import event, exc
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings

logger = logging.getLogger(__name__)

# MySQL client "ssl-mode" values that mean "use TLS".
_SSL_ON = {"REQUIRED", "VERIFY_CA", "VERIFY_IDENTITY"}


class Base(DeclarativeBase):
    pass


def normalize_database_url(raw: str) -> tuple[URL, str | None]:
    """Accept provider URLs as-is (e.g. Aiven's `mysql://…?ssl-mode=REQUIRED`).

    Returns the URL rewritten for the async driver, plus the requested MySQL ssl-mode (if any).
    Plain `mysql://` would otherwise make SQLAlchemy load the MySQLdb driver, which isn't installed.
    """
    url = make_url(raw)
    if url.get_backend_name() in ("mysql", "mariadb"):
        query = dict(url.query)
        ssl_mode = None
        for key in ("ssl-mode", "ssl_mode", "sslmode"):
            if key in query:
                ssl_mode = str(query.pop(key)).upper()
        if str(query.get("ssl", "")).lower() in ("true", "1", "required"):
            query.pop("ssl")
            ssl_mode = ssl_mode or "REQUIRED"
        query.setdefault("charset", "utf8mb4")
        url = url.set(drivername="mysql+aiomysql", query=query)
        return url, ssl_mode
    return url, None


def _connect_args(url: URL, ssl_mode: str | None) -> dict:
    settings = get_settings()
    wants_ssl = settings.database_ssl or (ssl_mode in _SSL_ON)
    if not wants_ssl or url.get_backend_name() != "mysql":
        return {}
    # Env vars often store PEM newlines as a literal backslash-n; restore real newlines.
    ca = settings.database_ssl_ca.replace("\\n", "\n").strip()
    if ca:
        return {"ssl": ssl.create_default_context(cadata=ca)}
    context = ssl.create_default_context()
    if ssl_mode == "REQUIRED":
        # MySQL's ssl-mode=REQUIRED encrypts without verifying the server certificate. Providers like
        # Aiven use their own CA, so set DATABASE_SSL_CA to also verify the certificate.
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        logger.warning("Database TLS is encrypted but unverified; set DATABASE_SSL_CA to verify the server.")
    return {"ssl": context}


_url, _ssl_mode = normalize_database_url(get_settings().database_url)
engine = create_async_engine(_url, pool_recycle=1800, connect_args=_connect_args(_url, _ssl_mode))

# pool_pre_ping would check every pooled connection before each use: a full round trip, which
# costs ~0.4 s when the database is in another region (and a question uses the pool twice).
# Only connections idle for a while, the ones a network drop may have closed, are checked.
IDLE_PING_SECONDS = 60


@event.listens_for(engine.sync_engine, "connect")
@event.listens_for(engine.sync_engine, "checkin")
def _note_last_use(dbapi_connection, connection_record) -> None:
    connection_record.info["last_used"] = time.monotonic()


@event.listens_for(engine.sync_engine, "checkout")
def _ping_if_idle(dbapi_connection, connection_record, connection_proxy) -> None:
    if time.monotonic() - connection_record.info.get("last_used", 0.0) < IDLE_PING_SECONDS:
        return
    try:
        alive = engine.dialect.do_ping(dbapi_connection)
    except Exception as error:
        raise exc.DisconnectionError() from error  # the pool replaces it with a fresh connection
    if not alive:
        raise exc.DisconnectionError()

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def create_tables() -> None:
    from . import models  # noqa: F401  (registers tables on Base.metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

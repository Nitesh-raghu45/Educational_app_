import ssl

from sqlalchemy.ext.asyncio import create_async_engine

from app.db import _connect_args, normalize_database_url

AIVEN = "mysql://avnadmin:s3cret@mysql-edu-proj.aivencloud.com:12345/defaultdb?ssl-mode=REQUIRED"


def test_aiven_url_uses_async_driver_and_keeps_credentials():
    url, ssl_mode = normalize_database_url(AIVEN)
    assert url.drivername == "mysql+aiomysql"
    assert (url.username, url.password, url.host, url.port, url.database) == (
        "avnadmin", "s3cret", "mysql-edu-proj.aivencloud.com", 12345, "defaultdb"
    )
    assert ssl_mode == "REQUIRED"
    assert "ssl-mode" not in url.query and url.query["charset"] == "utf8mb4"


def test_aiven_url_creates_engine_without_mysqldb():
    url, ssl_mode = normalize_database_url(AIVEN)
    engine = create_async_engine(url, connect_args=_connect_args(url, ssl_mode))
    assert engine.dialect.driver == "aiomysql"


def test_ssl_mode_required_without_ca_encrypts_without_verifying():
    url, ssl_mode = normalize_database_url(AIVEN)
    context = _connect_args(url, ssl_mode)["ssl"]
    assert isinstance(context, ssl.SSLContext) and context.verify_mode == ssl.CERT_NONE


def test_already_async_url_and_sqlite_are_unchanged():
    url, ssl_mode = normalize_database_url("mysql+aiomysql://u:p@localhost:3306/edu?charset=utf8mb4")
    assert url.drivername == "mysql+aiomysql" and ssl_mode is None
    assert _connect_args(url, ssl_mode) == {}
    url, ssl_mode = normalize_database_url("sqlite+aiosqlite:///./dev.db")
    assert url.drivername == "sqlite+aiosqlite" and ssl_mode is None


async def test_pool_pre_ping_works_with_aiomysql():
    """Regression: SQLAlchemy 2.0.36 crashed on every reused MySQL connection with
    'AsyncAdapt_aiomysql_connection.ping() missing 1 required positional argument: reconnect'."""
    from sqlalchemy.dialects.mysql import aiomysql as dialect_module
    from sqlalchemy.util import greenlet_spawn

    class FakeAiomysqlConnection:
        pings = 0

        async def ping(self, reconnect=True):
            FakeAiomysqlConnection.pings += 1

    dialect = dialect_module.dialect(dbapi=dialect_module.dialect.import_dbapi())
    connection = dialect_module.AsyncAdapt_aiomysql_connection(dialect.loaded_dbapi, FakeAiomysqlConnection())
    assert await greenlet_spawn(dialect.do_ping, connection) is True
    assert FakeAiomysqlConnection.pings == 1

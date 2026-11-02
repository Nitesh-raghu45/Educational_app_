"""The MySQL driver can send raw bytes (uploaded files, figure images).

Tests run on SQLite, so this checks the MySQL driver directly: PyMySQL 1.2 removed a function
aiomysql 0.2.0 calls to escape bytes, which broke every upload in production.
"""

import aiomysql


def test_aiomysql_escapes_bytes():
    connection = aiomysql.Connection(host="unused")  # never connects
    assert connection.escape(b"\x00'ab").startswith("_binary'")

import asyncio

from app import storage


def test_a_file_in_use_is_removed_once_it_is_released(tmp_path, monkeypatch):
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF")
    monkeypatch.setattr(storage, "_RETRY_SECONDS", 0.01)
    real_unlink = type(path).unlink
    calls = {"n": 0}

    def locked_twice(self, missing_ok=False):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(32, "The process cannot access the file")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(type(path), "unlink", locked_twice)

    async def scenario():
        storage.delete(str(path))  # must not raise
        assert path.exists()
        for _ in range(100):
            if not path.exists():
                break
            await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert not path.exists()


def test_without_an_event_loop_a_locked_file_is_left_without_raising(tmp_path, monkeypatch):
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF")

    def locked(self, missing_ok=False):
        raise PermissionError(32, "in use")

    monkeypatch.setattr(type(path), "unlink", locked)
    storage.delete(str(path))

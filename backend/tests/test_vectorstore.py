from app.config import Settings
from app.services import vectorstore
from app.services.vectorstore import PineconeStore


class FlakyIndex:
    """Times out on the first upsert, then succeeds; records the timeouts it was given."""

    def __init__(self):
        self.calls = []

    def upsert(self, vectors, namespace, show_progress, _request_timeout):
        self.calls.append((len(vectors), _request_timeout))
        if len(self.calls) == 1:
            raise TimeoutError("read timed out")


def test_upsert_uses_timeout_small_batches_and_retries(monkeypatch):
    monkeypatch.setattr(vectorstore.time, "sleep", lambda s: None)
    store = PineconeStore(Settings(pinecone_api_key="x"))
    store._index = FlakyIndex()
    store.upsert("ns", [(f"id{i}", [0.0], {}) for i in range(120)])
    sizes = [n for n, _ in store._index.calls]
    assert sizes == [50, 50, 50, 20]  # first batch retried after the timeout
    assert all(timeout == 30 for _, timeout in store._index.calls)

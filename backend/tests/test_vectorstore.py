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


class QueryIndex:
    def __init__(self):
        self.kwargs = None

    def query(self, **kwargs):
        self.kwargs = kwargs
        return type("Result", (), {"matches": []})()


def test_query_is_limited_to_the_owner_and_their_documents():
    store = PineconeStore(Settings(pinecone_api_key="x"))
    store._index = QueryIndex()
    store.query("docs--m", [0.0], 5, ["user-a", "library"], ["doc-1"])
    assert store._index.kwargs["namespace"] == "docs--m"
    assert store._index.kwargs["filter"] == {"user_id": {"$in": ["user-a", "library"]}, "document_id": {"$in": ["doc-1"]}}


class MigrationIndex:
    """In-memory stand-in for the parts of a Pinecone index the namespace migration uses."""

    def __init__(self, namespaces):
        self.data = namespaces  # name -> {id: (values, metadata)}

    def describe_index_stats(self, _request_timeout):
        return type("Stats", (), {"namespaces": {name: {} for name in self.data}})()

    def list(self, namespace):
        ids = list(self.data.get(namespace, {}))
        for start in range(0, len(ids), 100):
            yield ids[start : start + 100]

    def fetch(self, ids, namespace, _request_timeout):
        vector = lambda values, meta: type("Vector", (), {"values": values, "metadata": meta})()
        found = {i: vector(*self.data[namespace][i]) for i in ids if i in self.data.get(namespace, {})}
        return type("Fetched", (), {"vectors": found})()

    def upsert(self, vectors, namespace, show_progress, _request_timeout):
        target = self.data.setdefault(namespace, {})
        for v in vectors:
            target[v["id"]] = (v["values"], v["metadata"])

    def delete(self, delete_all, namespace, _request_timeout):
        assert delete_all
        self.data.pop(namespace, None)


def test_per_student_namespaces_are_moved_into_shared_ones():
    a, b = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
    index = MigrationIndex({
        f"user-{a}--paraphrase-multilingual-minilm-l12-v2": {
            f"a-{i}": ([float(i)], {"document_id": "d1", "chunk_index": i, "page": 0}) for i in range(130)
        },
        f"user-{b}--paraphrase-multilingual-minilm-l12-v2": {"b-0": ([9.0], {"document_id": "d2", "chunk_index": 0, "page": 1})},
        f"user-{a}": {"old-0": ([1.0], {"document_id": "d0", "chunk_index": 0, "page": 0})},  # legacy model
        f"user-{b}": {},  # empty: just removed
        "docs--paraphrase-multilingual-minilm-l12-v2": {"new-0": ([2.0], {"user_id": b, "document_id": "d3"})},
        "something-else": {"x": ([0.0], {})},
    })
    store = PineconeStore(Settings(pinecone_api_key="x"))
    store._index = index

    assert store.migrate_user_namespaces() == 132
    assert set(index.data) == {
        "docs--paraphrase-multilingual-minilm-l12-v2", "docs--all-minilm-l6-v2", "something-else",
    }
    shared = index.data["docs--paraphrase-multilingual-minilm-l12-v2"]
    assert len(shared) == 132  # 130 + 1 moved, 1 already there
    assert shared["a-7"] == ([7.0], {"document_id": "d1", "chunk_index": 7, "page": 0, "user_id": a})
    assert shared["b-0"][1]["user_id"] == b
    assert index.data["docs--all-minilm-l6-v2"]["old-0"][1]["user_id"] == a
    assert store.migrate_user_namespaces() == 0  # nothing left to move

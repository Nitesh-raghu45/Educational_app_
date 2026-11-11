import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings

logger = logging.getLogger(__name__)

# Old layout: one namespace per student, "user-<id>" (legacy model) or "user-<id>--<model slug>".
_USER_NAMESPACE = re.compile(r"^user-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:--(.+))?$")


@dataclass
class Match:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_ready(description: Any) -> bool:
    status = getattr(description, "status", None) or {}
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


class PineconeStore:
    """Thin synchronous wrapper around a Pinecone serverless index (call via asyncio.to_thread)."""

    # Smaller requests (~300 KB of JSON each) are less likely to stall on slow uplinks.
    UPSERT_BATCH = 50
    DELETE_BATCH = 1000

    def __init__(self, settings: Settings):
        self._settings = settings
        self._index = None
        self._lock = threading.Lock()

    def _get_index(self):
        with self._lock:
            if self._index is not None:
                return self._index
            s = self._settings
            if not s.pinecone_api_key:
                raise RuntimeError("PINECONE_API_KEY is not set")
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=s.pinecone_api_key)
            if s.pinecone_index not in pc.list_indexes().names():
                pc.create_index(
                    name=s.pinecone_index,
                    dimension=s.embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
                )
                deadline = time.monotonic() + 120
                while not _is_ready(pc.describe_index(s.pinecone_index)):
                    if time.monotonic() > deadline:
                        raise RuntimeError("Pinecone index was not ready after 120s")
                    time.sleep(2)
            self._index = pc.Index(s.pinecone_index)
            return self._index

    def _call(self, fn, *, timeout: float, attempts: int, **kwargs):
        """Pinecone requests have no timeout by default, so one stalled connection could leave a
        document "processing" forever. Bound each request and retry transient failures."""
        for attempt in range(1, attempts + 1):
            try:
                return fn(_request_timeout=timeout, **kwargs)
            except Exception:
                if attempt == attempts:
                    raise
                logger.warning("Pinecone %s failed (attempt %d/%d); retrying", fn.__name__, attempt, attempts, exc_info=True)
                time.sleep(min(2**attempt, 8))

    def upsert(self, namespace: str, items: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        index = self._get_index()
        for start in range(0, len(items), self.UPSERT_BATCH):
            batch = items[start : start + self.UPSERT_BATCH]
            # Safe to retry: vector ids are fixed, so a repeated upsert just overwrites.
            self._call(
                index.upsert,
                timeout=30,
                attempts=3,
                vectors=[{"id": i, "values": v, "metadata": m} for i, v, m in batch],
                namespace=namespace,
                show_progress=False,
            )

    def query(
        self, namespace: str, vector: list[float], top_k: int, owners: list[str], document_ids: list[str]
    ) -> list[Match]:
        if not document_ids:
            return []
        result = self._call(
            self._get_index().query,
            timeout=10,
            attempts=2,
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            # Students share the namespace: user_id keeps each search to vectors the user may read.
            filter={"user_id": {"$in": owners}, "document_id": {"$in": document_ids}},
        )
        return [Match(m.id, float(m.score), dict(m.metadata or {})) for m in result.matches]

    def delete(self, namespace: str, ids: list[str]) -> None:
        index = self._get_index()
        for start in range(0, len(ids), self.DELETE_BATCH):
            self._call(index.delete, timeout=30, attempts=3, ids=ids[start : start + self.DELETE_BATCH], namespace=namespace)

    def migrate_user_namespaces(self) -> int:
        """One-time move from one namespace per student to one shared namespace per model.

        Copies each old "user-<id>[--<model>]" namespace's vectors (values unchanged, so nothing
        is re-embedded) into the shared namespace for the same model, adding the owner's user_id,
        then deletes the old namespace. Safe to re-run after an interruption: copies overwrite
        by id, and an old namespace is only deleted once fully copied. Returns vectors moved.
        """
        from .providers import LEGACY_EMBEDDING_MODEL, model_slug, shared_namespace

        index = self._get_index()
        stats = self._call(index.describe_index_stats, timeout=30, attempts=3)
        moved = 0
        for name in sorted(stats.namespaces or {}):
            match = _USER_NAMESPACE.match(name)
            if not match:
                continue
            user_id, slug = match.group(1), match.group(2) or model_slug(LEGACY_EMBEDDING_MODEL)
            target = shared_namespace(slug)
            count = 0
            for ids in index.list(namespace=name):  # pages of up to 100 ids
                fetched = self._call(index.fetch, timeout=30, attempts=3, ids=list(ids), namespace=name)
                items = [
                    (vid, list(v.values), {**(v.metadata or {}), "user_id": user_id})
                    for vid, v in (fetched.vectors or {}).items()
                ]
                self.upsert(target, items)
                count += len(items)
            self._call(index.delete, timeout=30, attempts=3, delete_all=True, namespace=name)
            logger.info("Moved %d vector(s) from Pinecone namespace %s to %s", count, name, target)
            moved += count
        return moved

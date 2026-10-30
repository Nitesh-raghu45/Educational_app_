import asyncio
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="edu-tests-"))
os.environ.update(
    DATABASE_URL=os.environ.get("TEST_DATABASE_URL", f"sqlite+aiosqlite:///{(_tmp / 'test.db').as_posix()}"),
    REDIS_URL="",
    INGEST_MODE="inline",
    STORAGE_DIR=str(_tmp / "uploads"),
    JWT_SECRET="test-secret",
    GROQ_API_KEY="",
    PINECONE_API_KEY="",
    RETRIEVAL_MIN_SCORE="0.1",
    # Tests never search the internet; test_web_images.py turns this on with a fake search.
    WEB_IMAGES_ENABLED="false",
    # Some tests ask many questions in a row; test_auth_limits.py covers the limiter itself.
    RATE_LIMIT_PER_MINUTE="1000",
)

import httpx  # noqa: E402
import pytest  # noqa: E402

from app import ratelimit  # noqa: E402
from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services import jobs, providers  # noqa: E402
from app.services.vectorstore import Match  # noqa: E402

DIM = 4096


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors."""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        vector = [0.0] * DIM
        for word in text.lower().split():
            word = "".join(ch for ch in word if ch.isalnum())
            if word:
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class FakeVectorStore:
    def __init__(self):
        self.namespaces: dict[str, dict[str, tuple[list[float], dict]]] = {}

    def upsert(self, namespace, items):
        ns = self.namespaces.setdefault(namespace, {})
        for vector_id, values, metadata in items:
            ns[vector_id] = (values, metadata)

    def query(self, namespace, vector, top_k, document_ids):
        scored = [
            Match(vid, sum(a * b for a, b in zip(vector, values)), meta)
            for vid, (values, meta) in self.namespaces.get(namespace, {}).items()
            if meta["document_id"] in document_ids
        ]
        return sorted(scored, key=lambda m: m.score, reverse=True)[:top_k]

    def delete(self, namespace, ids):
        for vector_id in ids:
            self.namespaces.get(namespace, {}).pop(vector_id, None)


class FakeLLM:
    def __init__(self):
        self.fail_next_stream = False
        self.stream_calls: list[list[dict]] = []

    async def stream(self, messages, model, **kwargs):
        self.stream_calls.append(messages)
        if self.fail_next_stream:
            self.fail_next_stream = False
            raise RuntimeError("upstream LLM error")
        for piece in ["Photosynthesis turns light ", "into chemical energy [1]."]:
            yield piece

    async def complete(self, messages, model, json_mode=False, **kwargs):
        if json_mode:
            return json.dumps(
                {
                    "questions": [
                        {
                            "question": "What does photosynthesis produce?",
                            "options": ["Glucose and oxygen", "Nitrogen", "Salt", "Iron"],
                            "answer_index": 0,
                            "explanation": "The source says plants make glucose and release oxygen.",
                            "source": 1,
                        },
                        {"question": "Invalid: bad source", "options": ["a", "b"], "answer_index": 0, "source": 99},
                    ]
                }
            )
        # Query rewrite / summary: echo the last line of the prompt.
        return messages[-1]["content"].splitlines()[-1].replace("Latest message: ", "")


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture(autouse=True)
async def setup(fake_llm):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    providers.override(embedder=FakeEmbedder(), store=FakeVectorStore(), llm=fake_llm)
    ratelimit._local.clear()  # every test starts with fresh sign-up / sign-in counters
    yield
    # Let background generation/ingestion tasks finish before the event loop closes.
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    providers.override()


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        yield c


async def register(client, email="student@example.com") -> dict:
    response = await client.post("/api/v1/auth/register", json={"email": email, "password": "password123"})
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        name, data = "message", None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if data is not None:
            events.append((name, data))
    return events

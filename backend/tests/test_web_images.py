"""Pictures from the web: when to search, the time limit, caching, answers and provider parsing."""

import asyncio
import json
import time

import httpx
import pytest

from app.config import get_settings
from app.services import web_images
from app.services.web_images import WebImage, image_intent, search_query

from .conftest import parse_sse, register
from .test_chat import ask, new_section, upload_ready_doc

HEART = WebImage(
    title="Heart",
    image_url="https://upload.wikimedia.org/heart.png",
    page_url="https://en.wikipedia.org/wiki/File:Heart.png",
    source="wikipedia.org",
    provider="wikipedia",
    description="Organ that pumps blood",
    width=800,
    height=600,
)


class FakeSearch:
    def __init__(self, images=(HEART,), delay=0.0):
        self.images, self.delay = list(images), delay
        self.calls: list[tuple[str, bool]] = []

    async def search(self, query, *, use_google):
        self.calls.append((query, use_google))
        await asyncio.sleep(self.delay)
        return self.images


@pytest.fixture
def search(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "web_images_enabled", True)
    monkeypatch.setattr(web_images, "_memory", {})
    fake = FakeSearch()
    web_images.override(fake)
    yield fake
    web_images.override(None)


def write_answer(fake_llm, text):
    async def stream(messages, model, **kwargs):
        fake_llm.stream_calls.append(messages)
        yield text

    fake_llm.stream = stream


def test_intent_and_query():
    assert image_intent("Show me a picture of the human heart") == "explicit"
    assert image_intent("mitochondria ki photo dikhao") == "explicit"
    assert image_intent("Explain the structure of the heart") == "helpful"
    assert image_intent("Define inflation") is None
    assert search_query("Show me a picture of the human heart") == "human heart"
    assert search_query("What does a neuron look like?") == "neuron"


async def test_answer_shows_web_picture_inline(client, fake_llm, search):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    write_answer(fake_llm, "Plants make glucose [1].\n\n[[web:1]]")

    events = parse_sse((await ask(client, headers, section_id, "Show me a picture of photosynthesis in plants")).text)

    prompt = fake_llm.stream_calls[-1][-1]["content"]
    assert "WEB IMAGES:" in prompt and '<image id="1" title="Heart" from="wikipedia.org"' in prompt
    citations = next(d for n, d in events if n == "citations")["citations"]
    assert [c["filename"] for c in citations] == ["biology.txt", "wikipedia.org"]
    assert citations[1]["web_image"]["image_url"] == HEART.image_url and citations[1]["web_image"]["n"] == 1
    # Google is only used when the student asks for a picture.
    assert search.calls == [("photosynthesis plants", True)]
    messages = (await client.get(f"/api/v1/sections/{section_id}/messages", headers=headers)).json()
    assert messages[1]["citations"][1]["web_image"]["title"] == "Heart"


async def test_unused_or_unneeded_pictures(client, fake_llm, search):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)

    # No picture would help: no search at all.
    await ask(client, headers, section_id, "What does photosynthesis produce?")
    assert search.calls == []
    assert "WEB IMAGES" not in fake_llm.stream_calls[-1][-1]["content"]

    # Offered but not shown by the model: not stored.
    events = parse_sse((await ask(client, headers, section_id, "Explain the structure of plants in photosynthesis")).text)
    assert search.calls == [("structure plants photosynthesis", False)]
    citations = next(d for n, d in events if n == "citations")["citations"]
    assert all("web_image" not in c for c in citations)


async def test_slow_search_never_delays_the_answer(client, fake_llm, search, monkeypatch):
    monkeypatch.setattr(get_settings(), "web_image_timeout", 0.1)
    search.delay = 1.0
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)

    started = time.monotonic()
    await ask(client, headers, section_id, "Show me a photo of how photosynthesis makes glucose in plants")
    assert time.monotonic() - started < 0.9
    assert "WEB IMAGES" not in fake_llm.stream_calls[-1][-1]["content"]
    # The late search finishes in the background and fills the cache for next time.
    await asyncio.gather(*web_images._running)
    assert await web_images.find_images("photosynthesis makes glucose plants", use_google=True) == [HEART]
    assert len(search.calls) == 1


async def test_failed_search_answers_normally(client, fake_llm, search):
    async def broken(query, *, use_google):
        raise httpx.ConnectError("offline")

    search.search = broken
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    events = parse_sse((await ask(client, headers, section_id, "Show me a picture of photosynthesis")).text)
    assert [n for n, _ in events][-1] == "done"


async def test_picture_asked_for_but_not_in_documents(client, fake_llm, search):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    events = parse_sse((await ask(client, headers, section_id, "Show me an image of a galaxy nebula")).text)
    text = "".join(d["text"] for n, d in events if n == "token")
    assert "couldn't find" in text and text.endswith("[[web:1]]")
    assert fake_llm.stream_calls == []
    citations = next(d for n, d in events if n == "citations")["citations"]
    assert citations[0]["web_image"]["title"] == "Heart"


async def test_results_are_cached(search):
    assert await web_images.find_images("human heart", use_google=False) == [HEART]
    assert await web_images.find_images("human heart", use_google=False) == [HEART]
    assert len(search.calls) == 1


async def test_provider_parsing(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "en.wikipedia.org":
            assert request.url.params["gsrsearch"] == "human heart"
            pages = [
                {"index": 2, "title": "Cardiac cycle", "description": "Heart events"},  # no image: skipped
                {
                    "index": 1,
                    "title": "Heart",
                    "description": "Organ",
                    "pageimage": "Diagram of the human heart.svg",
                    "thumbnail": {"source": "https://upload.wikimedia.org/x/800px-heart.png", "width": 800, "height": 700},
                },
            ]
            return httpx.Response(200, json={"query": {"pages": pages}})
        assert request.headers["X-API-KEY"] == "test-key"
        body = json.loads(request.content)
        assert body["q"] == "human heart"
        images = [
            {"title": "Insecure", "imageUrl": "http://example.com/a.jpg", "link": "http://example.com"},
            {
                "title": "Heart anatomy",
                "imageUrl": "https://cdn.example.org/heart.jpg",
                "imageWidth": 1200,
                "imageHeight": 900,
                "link": "https://example.org/heart",
                "domain": "example.org",
            },
        ]
        return httpx.Response(200, json={"images": images})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        web_images.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.setattr(get_settings(), "serper_api_key", "test-key")

    images = await web_images.WebImageSearch().search("human heart", use_google=True)
    assert [(i.title, i.provider) for i in images] == [("Heart", "wikipedia"), ("Heart anatomy", "google")]
    assert images[0].page_url == "https://en.wikipedia.org/wiki/File:Diagram%20of%20the%20human%20heart.svg"
    assert images[1].source == "example.org" and images[1].width == 1200

    # Without a request for a picture, only Wikipedia is asked.
    images = await web_images.WebImageSearch().search("human heart", use_google=False)
    assert [i.provider for i in images] == ["wikipedia"]

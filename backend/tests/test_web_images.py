"""Pictures from the web: when to search, the time limit, caching, answers and provider parsing."""

import asyncio
import json
import time

import httpx
import pytest

from app.config import get_settings
from app.services import web_images
from app.services.conversation import needs_rewrite
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
async def search(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "web_images_enabled", True)
    monkeypatch.setattr(web_images, "_memory", {})
    fake = FakeSearch()
    web_images.override(fake)
    yield fake
    # Searches that outlived their answer finish here, not in the next test's event loop.
    await asyncio.gather(*web_images._running, return_exceptions=True)
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
    # The Zigbee chat: web requests are picture requests, and filler words are dropped.
    assert image_intent("ok then findout on the internet") == "explicit"
    assert image_intent("search google for zigbee") == "explicit"
    assert image_intent("What is the internet of things?") is None
    assert search_query("ok get some images about zigbee so i can understand it") == "zigbee"
    assert search_query("show me the zigbee related images that is inside my document") == "zigbee"
    assert needs_rewrite("ok then findout on the internet")
    assert search_query("Zigbee-related images inside the document") == "zigbee"


def test_topic_prefers_the_question_own_words():
    # The rewrite of "…about zigbee so i can understand it" dragged in earlier answers' titles.
    rewritten = "Zigbee protocol stack official Zigbee logo ETRX357 Zigbee images"
    assert web_images.topic_for("ok get some images about zigbee so i can understand it", rewritten) == "zigbee"
    # No topic in the message itself: the rewrite supplies it.
    assert web_images.topic_for("ok then findout on the internet", "Zigbee images on the internet") == "zigbee"
    # "its structure" says what kind of picture, not of what.
    assert web_images.topic_for("show me a picture of its structure", "Picture of the human heart") == "human heart structure"
    assert len(web_images.topic_for("show me it", "one two three four five six seven").split()) == 5


async def test_asked_for_pictures_get_longer_to_arrive(search, monkeypatch):
    monkeypatch.setattr(get_settings(), "web_image_timeout", 0.05)
    monkeypatch.setattr(get_settings(), "web_image_timeout_requested", 1.0)
    search.delay = 0.2
    assert await web_images.collect(web_images.start_search("Show me a photo of a volcano", "")) == [HEART]
    web_images._memory.clear()
    assert await web_images.collect(web_images.start_search("Explain the structure of a volcano", "")) == []


def test_ranking_prefers_explanatory_pictures():
    def image(title, description=""):
        return WebImage(title, "https://x.org/a.png", "https://x.org/a", "x.org", "wikipedia", description)

    stack, alliance, logo = image("Zigbee stack"), image("Connectivity Standards Alliance", "Maintains the Zigbee standard"), image("Zigbee logo")
    assert web_images.relevance(stack, "zigbee") > web_images.relevance(alliance, "zigbee") > 0
    assert web_images.relevance(logo, "zigbee") == 0  # explains nothing
    assert web_images.relevance(logo, "zigbee logo") > 0  # unless a logo was asked for


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
    monkeypatch.setattr(get_settings(), "web_image_timeout_requested", 0.1)  # "Show me a photo…"
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


NEBULA = WebImage("Crab Nebula", "https://upload.wikimedia.org/crab.jpg", "https://commons.wikimedia.org/wiki/File:Crab.jpg", "wikimedia.org", "wikipedia")


async def test_picture_asked_for_but_not_in_documents(client, fake_llm, search):
    search.images = [HEART, NEBULA]
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    events = parse_sse((await ask(client, headers, section_id, "Show me an image of a galaxy nebula")).text)
    text = "".join(d["text"] for n, d in events if n == "token")
    # Only the picture whose title names the topic: no model is choosing here.
    assert "couldn't find" in text and text.endswith("[[web:2]]") and "[[web:1]]" not in text
    assert fake_llm.stream_calls == []
    citations = next(d for n, d in events if n == "citations")["citations"]
    assert [c["web_image"]["title"] for c in citations] == ["Crab Nebula"]

    # Nothing matching: no picture rather than an unrelated one.
    search.images = [HEART]
    web_images._memory.clear()
    events = parse_sse((await ask(client, headers, section_id, "Show me an image of a galaxy nebula")).text)
    assert "[[web:" not in "".join(d["text"] for n, d in events if n == "token")


async def test_asked_for_pictures_but_model_shows_none(client, fake_llm, search):
    """"No images in your document" and nothing else: the matching web pictures are added."""
    search.images = [HEART, NEBULA]
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    write_answer(fake_llm, "There are no photosynthesis images in your document [1].")

    events = parse_sse((await ask(client, headers, section_id, "Show me photosynthesis images from my document")).text)
    text = "".join(d["text"] for n, d in events if n == "token")
    # Neither title names photosynthesis: nothing is added rather than unrelated pictures.
    assert text == "There are no photosynthesis images in your document [1]."

    search.images = [WebImage("Photosynthesis diagram", "https://upload.wikimedia.org/p.png", "https://commons.wikimedia.org/wiki/File:P.png", "wikimedia.org", "wikipedia"), HEART]
    web_images._memory.clear()
    events = parse_sse((await ask(client, headers, section_id, "Show me photosynthesis images from my document")).text)
    text = "".join(d["text"] for n, d in events if n == "token")
    assert text.endswith("Here is a picture from the web that may help:\n\n[[web:1]]")
    citations = next(d for n, d in events if n == "citations")["citations"]
    assert [c["web_image"]["title"] for c in citations if "web_image" in c] == ["Photosynthesis diagram"]


async def test_find_it_on_the_internet_follow_up(client, fake_llm, search):
    """"ok then find out on the internet" names no topic: the rewrite supplies it from the chat."""
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    await ask(client, headers, section_id, "What does photosynthesis produce?")
    search.calls.clear()

    async def rewrite(messages, model, **kwargs):
        return "Find photosynthesis images on the internet"

    fake_llm.complete = rewrite
    await ask(client, headers, section_id, "ok then findout on the internet")
    assert search.calls == [("photosynthesis", True)]


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
        if request.url.host == "commons.wikimedia.org":
            assert request.url.params["gsrnamespace"] == "6"
            files = [
                {"index": 2, "title": "File:Heart diagram-en.svg",
                 "imageinfo": [{"thumburl": "https://upload.wikimedia.org/t/heart-diagram.png", "thumbwidth": 800, "thumbheight": 640, "mime": "image/svg+xml"}]},
                {"index": 1, "title": "File:Cardiology lecture.pdf",  # not a picture: skipped
                 "imageinfo": [{"thumburl": "https://upload.wikimedia.org/t/lecture.jpg", "mime": "application/pdf"}]},
                {"index": 3, "title": "File:Anatomy class 1900.jpg",  # matches nothing: dropped
                 "imageinfo": [{"thumburl": "https://upload.wikimedia.org/t/class.jpg", "thumbwidth": 800, "thumbheight": 500, "mime": "image/jpeg"}]},
            ]
            return httpx.Response(200, json={"query": {"pages": files}})
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
    assert [(i.title, i.source) for i in images] == [
        ("Heart", "wikipedia.org"),
        ("Heart diagram-en", "wikimedia.org"),
        ("Heart anatomy", "example.org"),
    ]  # "Anatomy class 1900" matches nothing, so it isn't offered at all
    assert images[0].page_url == "https://en.wikipedia.org/wiki/File:Diagram%20of%20the%20human%20heart.svg"
    assert images[1].page_url == "https://commons.wikimedia.org/wiki/File:Heart_diagram-en.svg"
    assert images[2].width == 1200

    # Without a request for a picture, Google isn't asked (it costs money per search).
    images = await web_images.WebImageSearch().search("human heart", use_google=False)
    assert "example.org" not in [i.source for i in images]

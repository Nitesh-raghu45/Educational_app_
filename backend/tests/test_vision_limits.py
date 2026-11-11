"""Vision rate limits: provider fallback (Groq -> Gemini), waiting out short limits, deferring work
past daily limits and finishing it later, saved page transcripts and figure captions from them."""

import asyncio
import json
import uuid

import httpx
import pytest
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db import SessionLocal
from app.main import retry_vision_once
from app.models import DocumentChunk, DocumentFigure, FigureScan, PageTranscript, VisionBacklog
from app.services import jobs, providers, stored_files, vision
from app.services.vision import GeminiVision, GroqVision, VisionChain, VisionQuotaExceeded, parse_retry_after

from .conftest import register
from .test_math_documents import LATEX_PAGE, math_pdf, upload_pdf  # noqa: F401  (math_pdf is a fixture)

ADMIN = "admin@example.com"


# ---------- retry times ----------


def test_retry_after_is_read_from_provider_messages():
    assert parse_retry_after("on tokens per day (TPD). Please try again in 19m30.72s. Need more") == pytest.approx(1170.72)
    assert parse_retry_after("Please try again in 2m59.28s.") == pytest.approx(179.28)
    assert parse_retry_after("try again in 1h2m3s") == 3723
    assert parse_retry_after("try again in 450ms") == 1.0
    assert parse_retry_after("anything", header="12") == 12
    assert parse_retry_after("no time given") == vision.DEFAULT_RETRY_SECONDS


# ---------- the provider chain ----------


class Provider:
    def __init__(self, name, replies):
        self.name, self.replies, self.calls = name, list(replies), 0

    async def _next(self):
        self.calls += 1
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def describe(self, image, prompt):
        return await self._next()

    async def transcribe(self, image, prompt):
        return await self._next()


async def test_the_next_provider_takes_over_at_a_daily_limit():
    groq = Provider("groq", [VisionQuotaExceeded(3600, "groq")])
    gemini = Provider("gemini", ["from gemini"])
    chain = VisionChain([groq, gemini])
    assert await chain.transcribe(b"img", "p") == "from gemini"
    assert await chain.transcribe(b"img", "p") == "from gemini"
    assert groq.calls == 1  # paused after its limit: not asked again
    assert chain.available()


async def test_work_is_deferred_when_every_provider_is_at_its_limit():
    chain = VisionChain([Provider("groq", [VisionQuotaExceeded(3600)]), Provider("gemini", [VisionQuotaExceeded(600)])])
    with pytest.raises(VisionQuotaExceeded) as caught:
        await chain.describe(b"img", "p")
    assert 590 < caught.value.retry_after <= 600 and not chain.available()


async def test_short_limits_are_waited_out(monkeypatch):
    waits = []

    async def no_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(vision.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(vision.time, "monotonic", lambda: 1000.0 + 20 * len(waits))  # time passes while waiting
    groq = Provider("groq", [VisionQuotaExceeded(15), "ok"])
    assert await VisionChain([groq]).describe(b"img", "p") == "ok"
    assert len(waits) == 1 and 0 < waits[0] <= 15


async def test_other_failures_try_the_next_provider_then_raise():
    chain = VisionChain([Provider("groq", [RuntimeError("over capacity")]), Provider("gemini", ["ok"])])
    assert await chain.describe(b"img", "p") == "ok"
    broken = VisionChain([Provider("groq", [RuntimeError("over capacity")])])
    with pytest.raises(RuntimeError):
        await broken.describe(b"img", "p")


async def test_groq_rate_limits_become_quota_errors():
    import groq

    class Client:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
                    raise groq.RateLimitError(
                        "Rate limit reached ... tokens per day (TPD) ... Please try again in 19m30.72s.", response=response, body=None
                    )

    provider = GroqVision(Settings(groq_api_key="x"))
    provider._client = Client()
    with pytest.raises(VisionQuotaExceeded) as caught:
        await provider.transcribe(b"img", "p")
    assert caught.value.retry_after == pytest.approx(1170.72)


async def test_gemini_requests_and_replies():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if body["generationConfig"].get("responseMimeType") == "application/json":
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"title": "T"}'}]}, "finishReason": "STOP"}]})
        if len(seen) == 2:
            return httpx.Response(429, json={"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "33s"}]}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "MAX_TOKENS"}]})

    gemini = GeminiVision(Settings(gemini_api_key="g-key"), "gemini-test", transport=httpx.MockTransport(handler))
    assert await gemini.describe(b"\xff\xd8img", "describe") == '{"title": "T"}'
    request = seen[0]
    assert request.headers["x-goog-api-key"] == "g-key" and "gemini-test:generateContent" in str(request.url)
    assert json.loads(request.content)["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/jpeg"
    with pytest.raises(VisionQuotaExceeded) as caught:
        await gemini.transcribe(b"img", "page")
    assert caught.value.retry_after == 33
    with pytest.raises(RuntimeError):  # cut off: the end of the page would be lost
        await gemini.transcribe(b"img", "page")


def test_the_chain_uses_the_configured_providers():
    settings = Settings(groq_api_key="a", gemini_api_key="b", gemini_vision_models=" m-1, m-2 ,m-1,")
    assert [p.name for p in vision.create_vision(settings).providers] == ["groq", "gemini:m-1", "gemini:m-2"]
    assert [p.name for p in vision.create_vision(Settings(groq_api_key="", gemini_api_key="")).providers] == []


async def test_each_gemini_model_is_tried_until_one_has_quota_left():
    first = Provider("gemini:flash", [VisionQuotaExceeded(15000)])
    second = Provider("gemini:flash-lite", [VisionQuotaExceeded(15000)])
    third = Provider("gemini:other-lite", ["page text"])
    chain = VisionChain([Provider("groq", [VisionQuotaExceeded(3000)]), first, second, third])
    assert await chain.transcribe(b"img", "p") == "page text"
    assert await chain.transcribe(b"img", "p") == "page text"
    assert (first.calls, second.calls, third.calls) == (1, 1, 2)


# ---------- deferring and finishing work ----------


class LimitedVision:
    """Every call fails with a daily limit until `limited` is switched off."""

    def __init__(self):
        self.limited = True
        self.transcribed = self.described = 0

    def available(self):
        return not self.limited

    async def transcribe(self, image, prompt):
        if self.limited:
            raise VisionQuotaExceeded(3600)
        self.transcribed += 1
        return LATEX_PAGE + "\n[Figure: Fig. 6.9 – triangle ABC with DE parallel to BC]"

    async def describe(self, image, prompt):
        if self.limited:
            raise VisionQuotaExceeded(3600)
        self.described += 1
        return json.dumps({"title": "A diagram", "description": "Described by the vision model.", "useful": True})


async def scan_and_backlog(doc_id):
    async with SessionLocal() as session:
        scan = await session.get(FigureScan, doc_id)
        return scan.status, await session.get(VisionBacklog, doc_id) is not None, await stored_files.has_copy(session, doc_id)


async def test_math_pages_and_figures_wait_for_the_limit_and_finish_later(client, fake_llm, math_pdf):  # noqa: F811
    fake = LimitedVision()
    providers.override(embedder=providers.embedder(), store=providers.vector_store(), llm=fake_llm, vision=fake)
    headers = await register(client)
    doc_id = await upload_pdf(client, headers, math_pdf)

    # Ready and answerable from its text layer, with the math pages and figures waiting.
    assert await scan_and_backlog(doc_id) == ("waiting", True, True)  # the file is kept for later
    listed = (await client.get("/api/v1/documents", headers=headers)).json()[0]
    assert listed["status"] == "ready" and listed["vision_pending"] is True and listed["figure_status"] == "waiting"
    assert await retry_vision_once() == 0  # still at the limit: nothing is queued

    fake.limited = False
    assert await retry_vision_once() == 1
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    assert await scan_and_backlog(doc_id) == ("done", False, False)  # finished: the copy is released
    async with SessionLocal() as session:
        texts = " ".join(await session.scalars(select(DocumentChunk.text).where(DocumentChunk.document_id == doc_id)))
        figure = await session.scalar(select(DocumentFigure).where(DocumentFigure.document_id == doc_id))
        saved = list(await session.scalars(select(PageTranscript.page).where(PageTranscript.document_id == doc_id)))
    assert r"\frac{AD}{DB}" in texts and saved == [1]
    # The figure on the transcribed page was captioned from its transcript: no vision call.
    assert figure.title.startswith("Fig. 6.9") and fake.described == 0


async def test_saved_page_transcripts_are_never_paid_for_twice(client, fake_llm, math_pdf, monkeypatch):  # noqa: F811
    monkeypatch.setattr(get_settings(), "admin_emails", ADMIN)
    from app.services import library

    monkeypatch.setattr(library, "_admin_cache", None)
    fake = LimitedVision()
    fake.limited = False
    providers.override(embedder=providers.embedder(), store=providers.vector_store(), llm=fake_llm, vision=fake)
    admin = await register(client, ADMIN)
    response = await client.post(
        "/api/v1/library", files={"file": (math_pdf.name, math_pdf.read_bytes(), "application/pdf")},
        data={"grade": "10", "subject": "Mathematics"}, headers=admin,
    )
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    book_id = response.json()["id"]
    assert fake.transcribed == 1

    again = await client.post(f"/api/v1/library/{book_id}/reprocess", headers=admin)
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    assert again.status_code == 200 and fake.transcribed == 1  # the saved transcript was reused
    student = await register(client, "student@example.com")
    assert (await client.post(f"/api/v1/library/{book_id}/reprocess", headers=student)).status_code == 403


async def test_figures_alone_wait_for_the_limit(client, fake_llm, monkeypatch):
    from app.services import ingestion

    from .test_figures import new_section, upload_image

    fake = LimitedVision()
    providers.override(embedder=providers.embedder(), store=providers.vector_store(), llm=fake_llm, vision=fake)
    headers = await register(client)
    doc_id = await upload_image(client, headers, await new_section(client, headers), monkeypatch)
    assert await scan_and_backlog(doc_id) == ("waiting", False, True)

    fake.limited = False
    monkeypatch.setattr(ingestion, "ocr_image", lambda image: "The water cycle.")
    assert await retry_vision_once() == 1
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    async with SessionLocal() as session:
        assert (await session.get(FigureScan, doc_id)).status == "done"
        assert await session.scalar(select(DocumentFigure.title).where(DocumentFigure.document_id == uuid.UUID(str(doc_id)))) == "A diagram"


def test_passing_problems_are_retried_later_but_bad_requests_are_not():
    import groq

    request = httpx.Request("POST", "https://api.example.com")
    assert vision.is_retryable(VisionQuotaExceeded(10))
    assert vision.is_retryable(TimeoutError())
    assert vision.is_retryable(httpx.ConnectError("down", request=request))
    assert vision.is_retryable(httpx.HTTPStatusError("busy", request=request, response=httpx.Response(503, request=request)))
    assert vision.is_retryable(groq.APITimeoutError(request=request))
    assert not vision.is_retryable(httpx.HTTPStatusError("bad", request=request, response=httpx.Response(400, request=request)))
    assert not vision.is_retryable(RuntimeError("page transcription was cut off"))


class SlowVision(LimitedVision):
    async def transcribe(self, image, prompt):
        raise TimeoutError  # e.g. waiting out per-minute limits took too long


async def test_a_page_that_times_out_is_left_for_later_not_lost(client, fake_llm, math_pdf):  # noqa: F811
    providers.override(embedder=providers.embedder(), store=providers.vector_store(), llm=fake_llm, vision=SlowVision())
    doc_id = await upload_pdf(client, await register(client), math_pdf)
    status, in_backlog, has_copy = await scan_and_backlog(doc_id)
    assert in_backlog and has_copy and status == "waiting"


async def test_deleting_a_document_while_its_figures_run_is_harmless(client, fake_llm, monkeypatch):
    from app.services import figures

    from .test_figures import new_section, upload_image

    providers.override(embedder=providers.embedder(), store=providers.vector_store(), llm=fake_llm, vision=LimitedVision())
    headers = await register(client)
    doc_id = await upload_image(client, headers, await new_section(client, headers), monkeypatch)
    assert (await client.delete(f"/api/v1/documents/{doc_id}", headers=headers)).status_code == 204
    async with SessionLocal() as session:
        await figures._set_scan(session, doc_id, "waiting")  # what a run still in progress would do
        assert await session.get(FigureScan, doc_id) is None

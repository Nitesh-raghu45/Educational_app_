"""Vision models: describe figures and transcribe math-heavy pages.

Free tiers cap how many tokens a day each provider allows (Groq: 200,000 for its vision model),
so several providers are chained: Groq first, then Google Gemini when a key is set. A provider
that reports a rate limit is paused for as long as it asks. Short waits (per-minute limits) are
waited out; when every provider is paused for longer (daily limits), VisionQuotaExceeded tells
the caller to defer the work, and the background retry job (main.retry_vision_backlog) finishes
it once the quota is back.
"""

import asyncio
import base64
import logging
import re
import time
from typing import Protocol

import httpx

from ..config import Settings

logger = logging.getLogger(__name__)

# Waits up to this long are waited out (per-minute limits); longer ones defer the work.
SHORT_WAIT_SECONDS = 90
# When a provider gives no retry time: per-minute limits clear in about this long.
DEFAULT_RETRY_SECONDS = 60


class VisionQuotaExceeded(Exception):
    """No vision provider can take requests for `retry_after` seconds (rate or daily limits)."""

    def __init__(self, retry_after: float, provider: str = "vision"):
        super().__init__(f"{provider}: rate limit reached, retry in {retry_after:.0f}s")
        self.retry_after = retry_after
        self.provider = provider


_DURATION_RE = re.compile(r"try again in ((?:[\d.]+h)?(?:[\d.]+m(?!s))?(?:[\d.]+s)?(?:[\d.]+ms)?)", re.IGNORECASE)


def parse_retry_after(message: str, header: str | None = None) -> float:
    """Seconds to wait, from a retry-after header or text like "Please try again in 19m30.72s"."""
    if header:
        try:
            return max(float(header), 1.0)
        except ValueError:
            pass
    match = _DURATION_RE.search(message or "")
    if match and match.group(1):
        seconds = 0.0
        for amount, unit in re.findall(r"([\d.]+)(ms|h|m|s)", match.group(1)):
            seconds += float(amount) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
        return max(seconds, 1.0)
    return DEFAULT_RETRY_SECONDS


def is_retryable(exc: BaseException) -> bool:
    """A passing problem worth trying again later: rate limits, timeouts, network errors and
    server errors (5xx). Not a bad request or a reply cut off at its length limit."""
    if isinstance(exc, (VisionQuotaExceeded, TimeoutError, asyncio.TimeoutError, httpx.TransportError)):
        return True
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) or getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 429, 498) or status >= 500  # 498: Groq over capacity
    try:
        import groq

        return isinstance(exc, groq.APIConnectionError)  # includes groq.APITimeoutError
    except ImportError:  # pragma: no cover
        return False


class VisionProvider(Protocol):
    name: str

    async def describe(self, image_jpeg: bytes, prompt: str) -> str: ...
    async def transcribe(self, image_jpeg: bytes, prompt: str) -> str: ...


class GroqVision:
    name = "groq"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq

            # No retries in the client: VisionChain handles rate limits (it can switch to
            # Gemini instead of waiting), and passing errors defer the work (is_retryable).
            self._client = AsyncGroq(api_key=self._settings.groq_api_key, max_retries=0)
        return self._client

    async def _create(self, image_jpeg: bytes, prompt: str, **options):
        import groq

        url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}
        ]
        try:
            return await self._get_client().chat.completions.create(
                model=self._settings.groq_vision_model, messages=messages, reasoning_effort="none", **options
            )
        except groq.RateLimitError as exc:
            header = exc.response.headers.get("retry-after") if exc.response is not None else None
            raise VisionQuotaExceeded(parse_retry_after(str(exc), header), self.name) from exc

    async def describe(self, image_jpeg: bytes, prompt: str) -> str:
        # Describing what is visible needs no long thinking; this keeps captions ~1-2 s.
        response = await self._create(
            image_jpeg, prompt, temperature=0.1, max_completion_tokens=700, response_format={"type": "json_object"}
        )
        return response.choices[0].message.content or ""

    async def transcribe(self, image_jpeg: bytes, prompt: str) -> str:
        """Plain-text transcription of a page picture (a full page can need ~1,500 tokens)."""
        response = await self._create(image_jpeg, prompt, temperature=0, max_completion_tokens=3000)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("page transcription was cut off")  # the end of the page would be lost
        return choice.message.content or ""


class GeminiVision:
    """One Google Gemini model (generativelanguage API), used when Groq's limits are reached.
    Each model is its own provider in the chain, since each has its own free daily limit."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, settings: Settings, model: str, transport: httpx.AsyncBaseTransport | None = None):
        self._settings = settings
        self.model = model
        self.name = f"gemini:{model}"
        self._transport = transport

    async def _generate(self, image_jpeg: bytes, prompt: str, *, json_mode: bool, max_tokens: int) -> str:
        config: dict = {"temperature": 0.1 if json_mode else 0, "maxOutputTokens": max_tokens}
        if json_mode:
            config["responseMimeType"] = "application/json"
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_jpeg).decode()}},
                    ]
                }
            ],
            "generationConfig": config,
        }
        async with httpx.AsyncClient(transport=self._transport, timeout=120) as client:
            response = await client.post(
                self.URL.format(model=self.model),
                json=body,
                headers={"x-goog-api-key": self._settings.gemini_api_key},
            )
        if response.status_code == 429:
            raise VisionQuotaExceeded(_gemini_retry_after(response), self.name)
        response.raise_for_status()
        candidate = (response.json().get("candidates") or [{}])[0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise RuntimeError("Gemini reply was cut off")
        parts = (candidate.get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    async def describe(self, image_jpeg: bytes, prompt: str) -> str:
        return await self._generate(image_jpeg, prompt, json_mode=True, max_tokens=2048)

    async def transcribe(self, image_jpeg: bytes, prompt: str) -> str:
        # Room for the model's own thinking tokens, which count towards the output limit.
        return await self._generate(image_jpeg, prompt, json_mode=False, max_tokens=8192)


def _gemini_retry_after(response: httpx.Response) -> float:
    try:
        for detail in response.json().get("error", {}).get("details", []):
            if str(detail.get("@type", "")).endswith("RetryInfo") and detail.get("retryDelay"):
                return max(float(str(detail["retryDelay"]).rstrip("s")), 1.0)
    except (ValueError, AttributeError):
        pass
    return parse_retry_after(response.text, response.headers.get("retry-after"))


class VisionChain:
    """Tries each provider in order, skipping ones paused by a rate limit."""

    def __init__(self, providers: list[VisionProvider]):
        self.providers = providers
        self._paused_until: dict[str, float] = {}

    def wait_seconds(self) -> float:
        """0 when some provider can be called now, else how long until one can."""
        if not self.providers:
            return float("inf")
        now = time.monotonic()
        return max(0.0, min(self._paused_until.get(p.name, 0.0) for p in self.providers) - now)

    def available(self) -> bool:
        return self.wait_seconds() <= SHORT_WAIT_SECONDS

    async def _run(self, method: str, image_jpeg: bytes, prompt: str) -> str:
        if not self.providers:
            raise RuntimeError("No vision model is configured (set GROQ_API_KEY or GEMINI_API_KEY)")
        error: Exception | None = None
        for _ in range(4):
            for provider in self.providers:
                if self._paused_until.get(provider.name, 0.0) > time.monotonic():
                    continue
                try:
                    return await getattr(provider, method)(image_jpeg, prompt)
                except VisionQuotaExceeded as exc:
                    self._paused_until[provider.name] = time.monotonic() + exc.retry_after
                    logger.warning("Vision provider %s paused for %.0fs (rate limit)", provider.name, exc.retry_after)
                except Exception as exc:  # e.g. over capacity: the next provider may still answer
                    error = exc
                    logger.warning("Vision provider %s failed: %s", provider.name, exc)
            now = time.monotonic()
            paused = [until - now for until in self._paused_until.values() if until > now]
            if not paused:
                raise error or RuntimeError("vision request failed")  # failed for another reason
            # Some provider is only rate-limited: wait for it, or defer the work if that's long.
            wait = min(paused)
            if wait > SHORT_WAIT_SECONDS:
                raise VisionQuotaExceeded(wait)
            await asyncio.sleep(wait)
        if error is not None:
            raise error
        raise VisionQuotaExceeded(self.wait_seconds())

    async def describe(self, image_jpeg: bytes, prompt: str) -> str:
        return await self._run("describe", image_jpeg, prompt)

    async def transcribe(self, image_jpeg: bytes, prompt: str) -> str:
        return await self._run("transcribe", image_jpeg, prompt)


def create_vision(settings: Settings) -> VisionChain:
    providers: list[VisionProvider] = []
    if settings.groq_api_key:
        providers.append(GroqVision(settings))
    if settings.gemini_api_key:
        providers.extend(GeminiVision(settings, model) for model in settings.gemini_model_list)
    return VisionChain(providers)

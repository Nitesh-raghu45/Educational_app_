from typing import Any, AsyncIterator

from ..config import Settings

# Rate limited (429), Groq's capacity limit (498) or overloaded (5xx): another model may still answer.
_BUSY_STATUSES = {429, 498, 500, 502, 503}


class ModelBusy(Exception):
    """The model can't take this request right now; a different model might."""


class GroqLLM:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set")
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self._settings.groq_api_key)
        return self._client

    def _reasoning_options(self, model: str, effort: str | None = None) -> dict[str, Any]:
        """gpt-oss models think before answering. Keep that out of the reply and bound its effort."""
        if not model.startswith("openai/gpt-oss"):
            return {}
        effort = effort or (
            self._settings.groq_answer_reasoning_effort
            if model == self._settings.groq_answer_model
            else self._settings.groq_fast_reasoning_effort
        )
        return {"reasoning_effort": effort, "include_reasoning": False}

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        retry: bool = True,
    ) -> AsyncIterator[str]:
        """Streams the reply. With retry=False a busy model raises ModelBusy at once, instead of
        the client waiting out the rate limit (often 10+ s), so the caller can switch models."""
        from groq import APIStatusError

        client = self._get_client() if retry else self._get_client().with_options(max_retries=0)
        try:
            # max_completion_tokens also covers reasoning tokens, so it is set well above the answer length.
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stream=True,
                **self._reasoning_options(model, reasoning_effort),
            )
        except APIStatusError as exc:
            if exc.status_code in _BUSY_STATUSES:
                raise ModelBusy(f"{model}: HTTP {exc.status_code}") from exc
            raise
        async for chunk in stream:
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                yield delta

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        extra: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = await self._get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **self._reasoning_options(model),
            **extra,
        )
        return response.choices[0].message.content or ""


class GroqVision:
    """Describes figure images with a Groq vision model, for search and for the answer model."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None

    async def describe(self, image_jpeg: bytes, prompt: str) -> str:
        import base64

        if self._client is None:
            if not self._settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set")
            from groq import AsyncGroq

            # Figures are described in the background, so waiting out Groq's per-minute limit
            # (~20 s per retry) is fine; giving up after 2 retries lost figures in long PDFs.
            self._client = AsyncGroq(api_key=self._settings.groq_api_key, max_retries=8)
        url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()
        response = await self._client.chat.completions.create(
            model=self._settings.groq_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}],
                }
            ],
            temperature=0.1,
            max_completion_tokens=700,
            response_format={"type": "json_object"},
            # Describing what is visible needs no long thinking; this keeps captions ~1-2 s.
            reasoning_effort="none",
        )
        return response.choices[0].message.content or ""

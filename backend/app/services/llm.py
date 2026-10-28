from typing import Any, AsyncIterator

from ..config import Settings


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

    def _reasoning_options(self, model: str) -> dict[str, Any]:
        """gpt-oss models think before answering. Keep that out of the reply and bound its effort."""
        if not model.startswith("openai/gpt-oss"):
            return {}
        effort = (
            self._settings.groq_answer_reasoning_effort
            if model == self._settings.groq_answer_model
            else self._settings.groq_fast_reasoning_effort
        )
        return {"reasoning_effort": effort, "include_reasoning": False}

    async def stream(
        self, messages: list[dict[str, str]], model: str, *, temperature: float = 0.2, max_tokens: int = 4096
    ) -> AsyncIterator[str]:
        # max_completion_tokens also covers reasoning tokens, so it is set well above the answer length.
        stream = await self._get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
            **self._reasoning_options(model),
        )
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

            self._client = AsyncGroq(api_key=self._settings.groq_api_key)
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

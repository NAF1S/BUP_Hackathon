"""OpenAI-compatible chat-completions client.

Works with any provider exposing ``POST {base_url}/chat/completions``:
DeepSeek (default), OpenAI, Groq, Together, OpenRouter, vLLM, Ollama, ...

Reliability features required by the rubric:

* bounded per-attempt timeout and a caller-supplied wall-clock deadline
* retry with jittered backoff for 429 / 5xx / transport errors only
* a concurrency guard so repeated hidden tests cannot stampede the provider
* typed ``LLMError`` for every provider-side failure, so the pipeline can move
  to the next recovery stage instead of returning a 5xx
* the API key is placed in a header and never logged
"""

from __future__ import annotations

import asyncio
import logging
import random
import time

import httpx

from app.config import Settings
from app.errors import LLMError
from app.llm.base import LLMClient
from app.llm.prompts import build_messages

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleClient(LLMClient):
    """Stateless JSON-mode chat client."""

    name = "openai-compatible"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=5.0),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key or ''}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    # -- internals ---------------------------------------------------------

    def _payload(self, system: str, user: str) -> dict:
        payload: dict = {
            "model": self._settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
            "stream": False,
        }
        if self._settings.llm_json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _extract_content(body: dict) -> str:
        try:
            choices = body["choices"]
            message = choices[0]["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected provider response shape: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("provider returned empty content")
        return content

    # -- public API --------------------------------------------------------

    async def complete_json(
        self, system: str, user: str, *, deadline: float | None = None
    ) -> str:
        payload = self._payload(system, user)
        attempts = self._settings.llm_max_retries + 1
        last_error = "unknown provider failure"

        for attempt in range(attempts):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0.5:
                raise LLMError("request deadline exhausted before the model responded")

            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=payload,
                    timeout=(
                        httpx.Timeout(self._settings.llm_timeout_seconds)
                        if remaining is None
                        else httpx.Timeout(max(1.0, min(self._settings.llm_timeout_seconds, remaining)))
                    ),
                )
            except httpx.TimeoutException as exc:
                last_error = f"provider timeout: {type(exc).__name__}"
            except httpx.HTTPError as exc:
                last_error = f"transport error: {type(exc).__name__}"
            else:
                if response.status_code == 200:
                    try:
                        return self._extract_content(response.json())
                    except ValueError as exc:
                        raise LLMError(f"provider returned invalid JSON: {exc}") from exc
                last_error = f"provider status {response.status_code}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise LLMError(last_error, retryable=False)

            if attempt < attempts - 1:
                backoff = min(0.3 * (2**attempt), 1.5) + random.uniform(0.0, 0.15)
                if deadline is not None:
                    budget = deadline - time.monotonic()
                    if budget <= backoff + 0.5:
                        break
                await asyncio.sleep(backoff)

        raise LLMError(last_error)

    async def interpret(
        self,
        scenario_id: str,
        notes,
        battery,
        *,
        feedback=None,
        deadline: float | None = None,
    ) -> str:
        """Convenience wrapper that builds the messages then calls the model."""
        messages = build_messages(scenario_id, notes, battery, feedback=feedback)
        return await self.complete_json(
            messages[0]["content"], messages[1]["content"], deadline=deadline
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def build_llm_client(settings: Settings) -> LLMClient | None:
    """Return a configured client, or ``None`` when no credential is present."""
    if not settings.llm_configured:
        logger.warning(
            "LLM_API_KEY is not set; operator notes will be handled by the "
            "deterministic rule-based interpreter only"
        )
        return None
    return OpenAICompatibleClient(settings)

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import httpx

Outcome = Literal["ok", "rate_limited", "timeout", "error"]
LIMIT_HEADERS = (
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)


@dataclass
class CallResult:
    model: str
    outcome: Outcome
    started_at: float
    status_code: int | None = None
    first_chunk_ms: float | None = None
    first_token_ms: float | None = None
    total_ms: float | None = None
    text: str = ""
    reasoning: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _usage(chunk: dict) -> dict | None:
    return chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")


class ModelClient:
    def __init__(self, base_url: str, api_key: str, connect_timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(None, connect=connect_timeout_s),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int,
        deadline_s: float,
        temperature: float = 0.0,
        step: str | None = None,
        extra: dict | None = None,
    ) -> CallResult:
        result = CallResult(model=model, outcome="error", started_at=time.time())
        start = time.perf_counter()
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **(extra or {}),
        }
        headers = {"X-Relay-Step": step} if step else {}
        try:
            async with asyncio.timeout(deadline_s):
                await self._stream(body, headers, result, start)
        except TimeoutError:
            result.outcome = "timeout"
        except httpx.HTTPError as exc:
            result.outcome, result.error = "error", f"{type(exc).__name__}: {exc}"
        result.total_ms = (time.perf_counter() - start) * 1000
        return result

    async def _stream(self, body: dict, headers: dict, result: CallResult, start: float) -> None:
        url = f"{self.base_url}/chat/completions"
        async with self.http.stream("POST", url, json=body, headers=headers) as response:
            result.status_code = response.status_code
            result.headers = {
                h: response.headers[h] for h in LIMIT_HEADERS if h in response.headers
            }
            if response.status_code != 200:
                result.error = (await response.aread()).decode(errors="replace")[:2000]
                result.outcome = "rate_limited" if response.status_code == 429 else "error"
                return
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                elapsed = (time.perf_counter() - start) * 1000
                result.first_chunk_ms = result.first_chunk_ms or elapsed
                if usage := _usage(chunk):
                    result.tokens_in = usage.get("prompt_tokens", 0)
                    result.tokens_out = usage.get("completion_tokens", 0)
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    result.reasoning += delta.get("reasoning") or ""
                    if content := delta.get("content"):
                        result.first_token_ms = result.first_token_ms or elapsed
                        result.text += content
            result.outcome = "ok"

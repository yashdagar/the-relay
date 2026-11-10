import argparse
import asyncio
import json
import math
import random
import re
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

CALIBRATION_RUN = "results/20261111-0150-probe-latency"


@dataclass(frozen=True)
class ModelProfile:
    ttft_median_ms: float
    ttft_sigma: float
    decode_ms_per_token: float


PROFILES = {
    "openai/gpt-oss-120b": ModelProfile(481, 0.360, 1.12),
    "openai/gpt-oss-20b": ModelProfile(614, 0.679, 0.59),
    "qwen/qwen3.8-27b": ModelProfile(329, 0.237, 1.90),
}
DEFAULT_PROFILE = PROFILES["openai/gpt-oss-120b"]

PASSAGE = re.compile(r'<passage id="([^"]+)">\s*(.*?)\s*</passage>', re.S)
QUESTION = re.compile(r"<question>\s*(.*?)\s*</question>", re.S)
BOGUS_ID = "9999.99999v9:300:999"


@dataclass(frozen=True)
class StandInConfig:
    content: Literal["grounded", "uncited", "bogus_citation"] = "grounded"
    rpm: int | None = 30
    tpm: int | None = 8000
    rpd: int | None = 1000
    tpd: int | None = 200_000
    minute_s: float = 60.0
    day_s: float = 86_400.0
    ttft_ms: float | None = None
    fail_rate: float = 0.0
    slow_rate: float = 0.0
    slow_factor: float = 8.0
    overload_knee: int = 16
    overload_slope: float = 0.15
    chunk_tokens: int = 4
    seed: int | None = 0


class Bucket:
    def __init__(self, capacity: int, period_s: float):
        self.capacity = capacity
        self.rate = capacity / period_s if period_s else 0.0
        self.level = float(capacity)
        self.last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self.level = min(self.capacity, self.level + (now - self.last) * self.rate)
        self.last = now

    def wait_s(self, amount: float) -> float:
        self._refill()
        if self.level >= amount:
            return 0.0
        if self.rate == 0:
            return math.inf
        return (amount - self.level) / self.rate

    def take(self, amount: float) -> None:
        self.level -= amount

    def remaining(self) -> int:
        self._refill()
        return int(self.level)

    def reset_s(self) -> float:
        return (self.capacity - self.level) / self.rate if self.rate else math.inf


LIMIT_NAMES = {
    "rpd": ("requests per day (RPD)", "requests"),
    "tpd": ("tokens per day (TPD)", "tokens"),
    "rpm": ("requests per minute (RPM)", "requests"),
    "tpm": ("tokens per minute (TPM)", "tokens"),
}


class ModelState:
    def __init__(self, config: StandInConfig):
        self.buckets = {}
        for name, period in (
            ("rpm", config.minute_s),
            ("tpm", config.minute_s),
            ("rpd", config.day_s),
            ("tpd", config.day_s),
        ):
            capacity = getattr(config, name)
            if capacity is not None:
                self.buckets[name] = Bucket(capacity, period)

    def admit(self, tokens: int) -> tuple[str, float] | None:
        amounts = {name: 1 if name in ("rpd", "rpm") else tokens for name in self.buckets}
        for name in ("rpd", "tpd", "rpm", "tpm"):
            if name in self.buckets and (wait := self.buckets[name].wait_s(amounts[name])):
                return name, wait
        for name, amount in amounts.items():
            self.buckets[name].take(amount)
        return None

    def headers(self) -> dict[str, str]:
        out = {}
        if rpd := self.buckets.get("rpd"):
            out["x-ratelimit-limit-requests"] = str(rpd.capacity)
            out["x-ratelimit-remaining-requests"] = str(rpd.remaining())
            out["x-ratelimit-reset-requests"] = f"{min(rpd.reset_s(), 86_400):.3f}s"
        if tpm := self.buckets.get("tpm"):
            out["x-ratelimit-limit-tokens"] = str(tpm.capacity)
            out["x-ratelimit-remaining-tokens"] = str(tpm.remaining())
            out["x-ratelimit-reset-tokens"] = f"{min(tpm.reset_s(), 86_400):.3f}s"
        return out


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _first_sentence(text: str, max_words: int = 40) -> str:
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return " ".join(sentence.split()[:max_words]).rstrip(".") + "."


class StandIn:
    def __init__(self, config: StandInConfig):
        self.config = config
        self.random = random.Random(config.seed)
        self.models: dict[str, ModelState] = {}
        self.in_flight = 0
        self.app = Starlette(
            routes=[
                Route("/v1/chat/completions", self.chat, methods=["POST"]),
                Route("/v1/models", self.list_models, methods=["GET"]),
            ]
        )

    async def list_models(self, request: Request) -> Response:
        return JSONResponse({"data": [{"id": m} for m in PROFILES]})

    def _content(self, step: str, messages: list[dict]) -> str:
        prompt = "\n".join(m["content"] for m in messages)
        question = (QUESTION.search(prompt) or [None, ""])[1]
        if step == "understand":
            return json.dumps(
                {"in_scope": True, "query": question, "papers": [], "sub_questions": []}
            )
        if step == "check":
            return json.dumps({"supported": True, "answers_question": True, "problems": ""})
        if step == "draft":
            passages = PASSAGE.findall(prompt)
            if not passages:
                return "NOT_IN_PAPERS"
            chunk_id, text = passages[0]
            sentence = _first_sentence(text)
            if self.config.content == "uncited":
                return sentence
            cited = BOGUS_ID if self.config.content == "bogus_citation" else chunk_id
            return f"{sentence} [{cited}]"
        return "OK"

    def _ttft_s(self, model: str) -> float:
        profile = PROFILES.get(model, DEFAULT_PROFILE)
        median = self.config.ttft_ms or profile.ttft_median_ms
        ttft = self.random.lognormvariate(math.log(median), profile.ttft_sigma)
        overload = max(0, self.in_flight - self.config.overload_knee)
        ttft *= 1 + self.config.overload_slope * overload
        if self.random.random() < self.config.slow_rate:
            ttft *= self.config.slow_factor
        return ttft / 1000

    def _rate_limited(self, model: str, state: ModelState, limit: str, wait_s: float, tokens):
        description, kind = LIMIT_NAMES[limit]
        capacity = state.buckets[limit].capacity
        requested = 1 if kind == "requests" else tokens
        wait_text = f"{wait_s * 1000:.1f}ms" if wait_s < 1 else f"{wait_s:.2f}s"
        if math.isinf(wait_s):
            wait_text, retry_after = "a day", str(int(self.config.day_s))
        else:
            retry_after = str(max(1, math.ceil(wait_s)))
        message = (
            f"Rate limit reached for model `{model}` in organization `org_REDACTED` service tier "
            f"`on_demand` on {description}: Limit {capacity}, Used {capacity}, "
            f"Requested {requested}. Please try again in {wait_text}."
        )
        body = {"error": {"message": message, "type": kind, "code": "rate_limit_exceeded"}}
        return JSONResponse(
            body, status_code=429, headers={"retry-after": retry_after, **state.headers()}
        )

    async def chat(self, request: Request) -> Response:
        body = await request.json()
        model = body.get("model", "")
        state = self.models.setdefault(model, ModelState(self.config))
        prompt_tokens = sum(_estimate_tokens(m["content"]) for m in body["messages"])
        reserved = prompt_tokens + body.get("max_tokens", 1024)
        if limited := state.admit(reserved):
            return self._rate_limited(model, state, *limited, reserved)
        if self.random.random() < self.config.fail_rate:
            await asyncio.sleep(0.05)
            error = {"error": {"message": "Service Unavailable", "type": "server_error"}}
            return JSONResponse(error, status_code=503, headers=state.headers())

        step = request.headers.get("x-relay-step", "")
        text = self._content(step, body["messages"])
        ttft_s = self._ttft_s(model)
        decode_s = PROFILES.get(model, DEFAULT_PROFILE).decode_ms_per_token / 1000
        headers = state.headers()
        return StreamingResponse(
            self._stream(model, text, prompt_tokens, ttft_s, decode_s),
            media_type="text/event-stream",
            headers=headers,
        )

    async def _stream(self, model, text, prompt_tokens, ttft_s, decode_s):
        self.in_flight += 1
        try:
            await asyncio.sleep(ttft_s)
            words = re.findall(r"\S+\s*", text)
            step = self.config.chunk_tokens
            for i in range(0, len(words), step):
                delta = {"content": "".join(words[i : i + step])}
                yield _sse({"model": model, "choices": [{"index": 0, "delta": delta}]})
                await asyncio.sleep(decode_s * step)
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": _estimate_tokens(text)}
            yield _sse(
                {
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "x_groq": {"usage": usage},
                }
            )
            yield "data: [DONE]\n\n"
        finally:
            self.in_flight -= 1


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@contextmanager
def running(config: StandInConfig):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(StandIn(config).app, log_level="warning", lifespan="off")
    )
    thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[sock])))
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join()
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stand-in model service")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--content", default="grounded")
    args = parser.parse_args()
    app = StandIn(StandInConfig(content=args.content)).app
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

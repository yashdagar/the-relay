import asyncio
import json
import math
import random
import re
import time
from functools import cache

from relay.client import CallResult, ModelClient
from relay.config import Config
from relay.eval import CHUNK_ID, answer_body, is_grounded, numbers
from relay.governor import DAILY, Governor, parse_rate_limit
from relay.library import Chunk
from relay.prompts import check_messages, draft_messages, understand_messages
from relay.reply import Reply
from relay.retrieval import RetrievalSetting, Retriever, interleave

__all__ = ["Pipeline", "Reply"]

JSON_OBJECT = re.compile(r"\{.*\}", re.S)
CITED_ID = re.compile(rf"[\[【]?\s*({CHUNK_ID.pattern})\s*[\]】]?")
NOT_IN_PAPERS = "NOT_IN_PAPERS"
TRUNCATED_FEEDBACK = "it was cut off by the length limit. Answer in at most three short sentences."
FAILURE_STATUS = {
    "exhausted": "rate_limited",
    "rate_limited": "rate_limited",
    "timeout": "timeout",
    "error": "error",
}

REFUSALS = {
    "out_of_scope": (
        "I can only answer questions about three papers: Attention Is All You Need, Moshi, "
        "and Dynamic Chunking for End-to-End Hierarchical Sequence Modeling. This question "
        "is outside them."
    ),
    "not_in_papers": "The papers do not say. I could not find this in any of the three papers.",
    "check_failed": (
        "I could not write an answer that I could verify against the papers, so I am not "
        "sending one. The papers may not cover this."
    ),
}


@cache
def _retriever(setting: RetrievalSetting) -> Retriever:
    return Retriever(setting)


def _parse_json(text: str) -> dict:
    match = JSON_OBJECT.search(text)
    try:
        return json.loads(match.group(0)) if match else {}
    except ValueError:
        return {}


def _estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    return sum(len(m["content"]) for m in messages) // 4 + max_tokens


def grounding_feedback(candidate: Reply, cited_ids: set[str]) -> str:
    if not cited_ids:
        return "it cited no passage. End every factual sentence with a passage id."
    unknown = sorted(cited_ids - {c.id for c in candidate.retrieved})
    if unknown:
        return f"it cited passages that were not provided: {', '.join(unknown)}."
    supported = set().union(*(numbers(c.text) for c in candidate.citations))
    missing = sorted(numbers(answer_body(candidate.text)) - supported)
    return (
        f"it stated {', '.join(missing)}, which its cited passages do not contain. Remove "
        "numbers you computed or recalled, and state only numbers written in the passages."
    )


def _display(text: str, chunks: list[Chunk]) -> str:
    by_id = {c.id: c for c in chunks}
    return CITED_ID.sub(
        lambda m: f"[{by_id[m.group(1)].cite()[1:-1]}]" if m.group(1) in by_id else m.group(0),
        text,
    )


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.client = ModelClient(config.base_url, config.api_key)
        self.governor = Governor(
            config.limits, minute_s=config.governor_minute_s, day_s=config.governor_day_s
        )
        self.retriever = _retriever(config.retrieval)
        self.active = 0

    async def aclose(self) -> None:
        await self.client.aclose()

    async def answer(self, question: str) -> Reply:
        request = _Request(self, question, queue_depth=self.active)
        self.active += 1
        try:
            return await request.run()
        finally:
            self.active -= 1


class _Request:
    def __init__(self, pipeline: Pipeline, question: str, queue_depth: int):
        self.pipeline = pipeline
        self.config = pipeline.config
        self.governor = pipeline.governor
        self.question = question
        self.queue_depth = queue_depth
        self.start = time.monotonic()
        self.deadline = self.start + self.config.deadline_ms / 1000
        self.retries = 0
        self.model_calls = 0
        self.drafts = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.failure: str | None = None
        self.steps: list[dict] = []

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.start) * 1000

    def remaining_s(self) -> float:
        return self.deadline - time.monotonic()

    def reply(self, kind: str, text: str, **fields) -> Reply:
        total = self.elapsed_ms()
        return Reply(
            kind=kind,
            text=text,
            total_ms=total,
            first_token_ms=fields.pop("first_token_ms", total),
            retries=self.retries,
            model_calls=self.model_calls,
            drafts=self.drafts,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            queue_depth=self.queue_depth,
            steps=self.steps,
            **fields,
        )

    def refusal(self, reason: str, retrieved: list[Chunk] = ()) -> Reply:
        return self.reply("refusal", REFUSALS[reason], reason=reason, retrieved=list(retrieved))

    async def run(self) -> Reply:
        sub_questions = await self.understand()
        if sub_questions is None:
            return self.refusal("out_of_scope")
        chunks, top_score = self.retrieve(sub_questions)
        if top_score < self.config.refuse_below_score:
            return self.refusal("not_in_papers", chunks)
        return await self.draft_and_check(chunks)

    async def understand(self) -> list[str] | None:
        result = await self.call(
            "understand",
            self.config.understand_model,
            understand_messages(self.question),
            self.config.understand_max_tokens,
            self.config.understand_timeout_ms,
        )
        if result is None:
            return []
        parsed = _parse_json(result.text)
        if parsed.get("in_scope") is False:
            return None
        subs = parsed.get("sub_questions") or []
        return [s for s in subs if isinstance(s, str)][: self.config.max_extra_retrievals]

    def retrieve(self, sub_questions: list[str]) -> tuple[list[Chunk], float]:
        retriever = self.pipeline.retriever
        hits = retriever.search(self.question)
        extra = [[h.chunk for h in retriever.search(sub)][:2] for sub in sub_questions]
        top_score = hits[0].score if hits else 0.0
        return interleave([h.chunk for h in hits], *extra), top_score

    def choose_draft(self, chunks: list[Chunk]) -> tuple[str, list[Chunk]] | None:
        config = self.config
        reserve_s = (config.draft_reserve_ms + config.check_reserve_ms) / 1000
        if self.governor.wait_s(config.check_model, 0) + reserve_s > self.remaining_s():
            return None
        for model, passages in (
            (config.draft_model, chunks),
            (config.lean_draft_model, chunks[: config.lean_top_k]),
        ):
            tokens = _estimate_tokens(
                draft_messages(self.question, passages, None), config.draft_max_tokens
            )
            if self.governor.wait_s(model, tokens) + reserve_s <= self.remaining_s():
                return model, passages
        if self.governor.daily_exhausted(config.draft_model):
            self.failure = "exhausted"
        return None

    async def draft_and_check(self, chunks: list[Chunk]) -> Reply:
        config = self.config
        feedback = None
        while self.drafts < config.max_drafts:
            choice = self.choose_draft(chunks)
            if choice is None:
                return self.degraded(chunks)
            model, passages = choice
            self.drafts += 1
            draft = await self.call(
                "draft",
                model,
                draft_messages(self.question, passages, feedback),
                config.draft_max_tokens,
                config.draft_timeout_ms,
            )
            if draft is None:
                return self.degraded(chunks)
            text = draft.text.strip()
            if text.startswith(NOT_IN_PAPERS):
                return self.refusal("not_in_papers", passages)
            if draft.finish_reason == "length":
                feedback = TRUNCATED_FEEDBACK
                continue
            cited_ids = set(CHUNK_ID.findall(text))
            cited = [c for c in passages if c.id in cited_ids]
            candidate = Reply(kind="answer", text=text, citations=cited, retrieved=passages)
            if not is_grounded(candidate) or len(cited) < len(cited_ids):
                feedback = grounding_feedback(candidate, cited_ids)
                continue
            check = await self.call(
                "check",
                config.check_model,
                check_messages(self.question, passages, text),
                config.check_max_tokens,
                config.check_timeout_ms,
            )
            if check is None:
                return self.degraded(chunks)
            verdict = _parse_json(check.text)
            if verdict.get("supported") is True and verdict.get("answers_question") is True:
                check_passed_ms = self.elapsed_ms()
                return self.reply(
                    "answer",
                    _display(text, passages),
                    citations=cited,
                    retrieved=passages,
                    check_passed_ms=check_passed_ms,
                    first_token_ms=self.elapsed_ms(),
                )
            feedback = verdict.get("problems") or "the checker found an unsupported claim."
        return self.refusal("check_failed", chunks)

    def degraded(self, chunks: list[Chunk]) -> Reply:
        failure = self.failure or "rate_limited"
        shown = chunks[:3]
        passages = "\n\n".join(f"{c.cite()} {' '.join(c.text.split()[:80])}..." for c in shown)
        if failure == "exhausted":
            reason = "budget_exhausted"
            wait_s = max(
                self.governor.exhausted_for_s(m)
                for m in (self.config.draft_model, self.config.lean_draft_model)
            )
            per_request_s = self.config.governor_day_s / min(
                lim.requests_per_day for lim in self.config.limits.values()
            )
            if math.isinf(wait_s):
                wait_s = per_request_s
            wait_min = max(1, math.ceil(max(wait_s, per_request_s) / 60))
            lead = (
                "The assistant has used up today's model budget, so it cannot write a checked "
                f"answer right now. The budget resets gradually; try again in about {wait_min} "
                "minutes."
            )
        else:
            reason = failure
            lead = (
                "The assistant is too busy to write a checked answer in time, and it will not "
                "send an unchecked one. Please try again in a minute."
            )
        text = (
            f"{lead}\n\nThese passages from the papers matched your question best. They are "
            f"quoted, not checked, and may not answer it:\n\n{passages}"
        )
        return self.reply(
            "degraded",
            text,
            reason=reason,
            status=FAILURE_STATUS[failure],
            retrieved=shown,
        )

    async def call(
        self, step: str, model: str, messages: list[dict], max_tokens: int, timeout_ms: int
    ) -> CallResult | None:
        config = self.config
        tokens = _estimate_tokens(messages, max_tokens)
        self.model_calls += 1
        backoff_s = 0.0
        for attempt in range(config.max_retries + 1):
            wait_start = time.monotonic()
            if backoff_s:
                await asyncio.sleep(min(backoff_s, max(0.0, self.remaining_s())))
            if self.remaining_s() <= 0:
                self.failure = "timeout"
                return None
            if not await self.governor.acquire(model, tokens, self.deadline):
                exhausted = self.governor.daily_exhausted(model)
                self.failure = "exhausted" if exhausted else "rate_limited"
                return None
            queued_ms = (time.monotonic() - wait_start) * 1000
            result = await self.pipeline.client.chat(
                model,
                messages,
                max_tokens=max_tokens,
                deadline_s=min(timeout_ms / 1000, self.remaining_s()),
                step=step,
                extra=config.extra(model),
            )
            self.trace(step, model, attempt, queued_ms, result)
            self.governor.observe(model, result.headers)
            self.tokens_in += result.tokens_in
            self.tokens_out += result.tokens_out
            if result.outcome == "ok":
                return result
            if result.outcome == "rate_limited":
                limit, wait_s = parse_rate_limit(result.error or "")
                if wait_s is None:
                    wait_s = float(result.headers.get("retry-after", 1))
                self.governor.penalize(model, limit, wait_s)
                self.failure = "exhausted" if limit in DAILY else "rate_limited"
                if limit in DAILY:
                    return None
            else:
                self.failure = result.outcome
            if self.retries >= config.max_retries:
                return None
            if result.outcome == "timeout" and self.remaining_s() < timeout_ms / 1000:
                return None
            self.retries += 1
            backoff_s = 0.0
            if result.outcome == "error":
                base_s = config.backoff_base_ms / 1000 * 2 ** (self.retries - 1)
                backoff_s = base_s * random.uniform(0.5, 1.5)
        return None

    def trace(self, step: str, model: str, attempt: int, queued_ms: float, result) -> None:
        self.steps.append(
            {
                "step": step,
                "model": model,
                "attempt": attempt,
                "started_ms": round(self.elapsed_ms() - result.total_ms, 1),
                "queued_ms": round(queued_ms, 1),
                "first_token_ms": result.first_token_ms and round(result.first_token_ms, 1),
                "duration_ms": round(result.total_ms, 1),
                "outcome": result.outcome,
                "status_code": result.status_code,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "text": (result.text or result.error or "")[:600],
            }
        )

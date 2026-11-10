import asyncio

import thresholds

QUESTION = "What value of warmup_steps was used in the Transformer's learning rate schedule?"


async def test_grounded_draft_is_sent_with_citations(standin_pipeline):
    reply = await standin_pipeline(content="grounded").answer(QUESTION)
    assert reply.kind == "answer" and reply.citations


async def test_first_token_is_sent_only_after_the_check_passes(standin_pipeline):
    reply = await standin_pipeline(content="grounded").answer(QUESTION)
    assert reply.first_token_ms >= reply.check_passed_ms


async def test_draft_without_citations_is_never_sent(standin_pipeline):
    reply = await standin_pipeline(content="uncited").answer(QUESTION)
    assert reply.kind != "answer"


async def test_draft_citing_a_passage_that_was_not_retrieved_is_never_sent(standin_pipeline):
    reply = await standin_pipeline(content="bogus_citation").answer(QUESTION)
    assert reply.kind != "answer"


async def test_failed_check_redrafts_once_then_refuses(standin_pipeline):
    pipeline = standin_pipeline(content="uncited")
    reply = await pipeline.answer(QUESTION)
    assert reply.reason == "check_failed" and reply.drafts == 2


async def test_model_calls_per_request_are_bounded(standin_pipeline):
    pipeline = standin_pipeline(content="uncited")
    reply = await pipeline.answer(QUESTION)
    assert reply.model_calls <= pipeline.config.max_model_calls


async def test_rate_limited_request_is_retried_and_succeeds(standin_pipeline):
    pipeline = standin_pipeline(content="grounded", rpm=1, minute_s=2.0)
    replies = await asyncio.gather(pipeline.answer(QUESTION), pipeline.answer(QUESTION))
    assert all(r.status == "ok" for r in replies)
    assert max(r.retries for r in replies) >= 1


async def test_failing_provider_gives_up_after_the_retry_limit(standin_pipeline):
    pipeline = standin_pipeline(content="grounded", fail_rate=1.0)
    reply = await pipeline.answer(QUESTION)
    assert reply.status == "error" and reply.retries <= pipeline.config.max_retries


async def test_failing_provider_never_produces_an_answer(standin_pipeline):
    reply = await standin_pipeline(content="grounded", fail_rate=1.0).answer(QUESTION)
    assert reply.kind != "answer"


async def test_slow_provider_is_abandoned_at_the_deadline(standin_pipeline):
    pipeline = standin_pipeline({"deadline_ms": 2000}, content="grounded", ttft_ms=20_000)
    reply = await pipeline.answer(QUESTION)
    assert reply.status == "timeout"
    assert reply.total_ms <= 2000 + thresholds.SYSTEM_DEADLINE_GRACE_MS


async def test_exhausted_daily_budget_is_reported_without_waiting(standin_pipeline):
    reply = await standin_pipeline(content="grounded", rpd=0).answer(QUESTION)
    assert reply.reason == "budget_exhausted"
    assert reply.total_ms <= thresholds.SYSTEM_BUDGET_EXHAUSTED_MAX_REPLY_MS


async def test_exhausted_daily_budget_reply_says_when_service_returns(standin_pipeline):
    reply = await standin_pipeline(content="grounded", rpd=0).answer(QUESTION)
    assert "reset" in reply.text.lower()


async def test_exhausted_daily_budget_reply_is_never_an_unchecked_answer(standin_pipeline):
    reply = await standin_pipeline(content="grounded", rpd=0).answer(QUESTION)
    assert reply.kind in {"degraded", "refusal"}

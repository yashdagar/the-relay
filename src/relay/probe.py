import argparse
import asyncio
import json
import os
import subprocess
from datetime import datetime

from dotenv import load_dotenv

from relay.client import ModelClient
from relay.library import ROOT

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b")
TINY = [{"role": "user", "content": "Reply with the single word OK."}]


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


async def limits(client: ModelClient) -> list[dict]:
    rows = []
    for model in CHAT_MODELS:
        result = await client.chat(model, TINY, max_tokens=64, deadline_s=60, step="probe-limits")
        rows.append(result.to_dict())
    return rows


LATENCY_MODELS = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b")
MODEL_EXTRA = {
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
}
LATENCY_CALLS_PER_MODEL = 20
LATENCY_SPACING_S = 22
DRAFT_MAX_TOKENS = 400
DRAFT_SYSTEM = (
    "You answer questions using only the numbered passages from three research papers. "
    "Cite every claim with the passage id in square brackets. If the passages do not "
    "answer the question, say that the papers do not say."
)
RPM_BURST_MODEL = "allam-2-7b"
RPM_BURST_SIZE = 40
RECOVERY_POLL_S = 5
RECOVERY_MAX_S = 120
TPM_BURST_MODEL = "openai/gpt-oss-20b"
TPM_BURST_SIZE = 4


def draft_messages(question: str) -> list[dict]:
    from relay.retrieval import DEFAULT_SETTING, Retriever

    hits = Retriever(DEFAULT_SETTING).search(question)
    passages = "\n\n".join(f"[{h.chunk.id}] {h.chunk.cite()}\n{h.chunk.text}" for h in hits)
    return [
        {"role": "system", "content": DRAFT_SYSTEM},
        {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {question}"},
    ]


def _tag(result, **labels) -> dict:
    return {**labels, **result.to_dict()}


async def latency(client: ModelClient) -> list[dict]:
    from relay.questions import QUESTIONS, expects_answer

    asked = [q for q in QUESTIONS if expects_answer(q)][:LATENCY_CALLS_PER_MODEL]
    prompts = [(q["id"], draft_messages(q["question"])) for q in asked]

    async def per_model(model: str) -> list[dict]:
        rows = []
        for i, (qid, messages) in enumerate(prompts):
            started = asyncio.get_running_loop().time()
            result = await client.chat(
                model,
                messages,
                max_tokens=DRAFT_MAX_TOKENS,
                deadline_s=60,
                step="probe-latency",
                extra=MODEL_EXTRA.get(model),
            )
            rows.append(_tag(result, seq=i, question_id=qid))
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0.0, LATENCY_SPACING_S - elapsed))
        return rows

    per = await asyncio.gather(*(per_model(m) for m in LATENCY_MODELS))
    return [row for rows in per for row in rows]


async def rpm(client: ModelClient) -> list[dict]:
    async def tiny(seq: int, label: str) -> dict:
        result = await client.chat(
            RPM_BURST_MODEL, TINY, max_tokens=8, deadline_s=60, step=f"probe-{label}"
        )
        return _tag(result, seq=seq, label=label)

    rows = list(await asyncio.gather(*(tiny(i, "burst") for i in range(RPM_BURST_SIZE))))
    loop = asyncio.get_running_loop()
    burst_end = loop.time()
    for seq in range(RPM_BURST_SIZE, RPM_BURST_SIZE + RECOVERY_MAX_S // RECOVERY_POLL_S):
        await asyncio.sleep(RECOVERY_POLL_S)
        row = await tiny(seq, "recovery")
        row["since_burst_s"] = loop.time() - burst_end
        rows.append(row)
        if row["outcome"] == "ok":
            break
    return rows


async def tpm(client: ModelClient) -> list[dict]:
    from relay.questions import QUESTIONS

    messages = draft_messages(QUESTIONS[0]["question"])
    rows = []
    for seq in range(TPM_BURST_SIZE):
        result = await client.chat(
            TPM_BURST_MODEL,
            messages,
            max_tokens=DRAFT_MAX_TOKENS,
            deadline_s=60,
            step="probe-tpm",
            extra=MODEL_EXTRA.get(TPM_BURST_MODEL),
        )
        rows.append(_tag(result, seq=seq))
    return rows


PHASES = {"limits": limits, "latency": latency, "rpm": rpm, "tpm": tpm}


async def _run(phase: str) -> None:
    load_dotenv(ROOT / ".env")
    client = ModelClient(GROQ_BASE_URL, os.environ["GROQ_API_KEY"])
    started = datetime.now()
    try:
        rows = await PHASES[phase](client)
    finally:
        await client.aclose()
    out = ROOT / "results" / f"{started:%Y%m%d-%H%M}-probe-{phase}"
    out.mkdir(parents=True)
    with (out / "calls.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    run_info = {
        "phase": phase,
        "commit": _commit(),
        "started": started.isoformat(),
        "calls": len(rows),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2))
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the real Groq service")
    parser.add_argument("phase", choices=sorted(PHASES))
    asyncio.run(_run(parser.parse_args().phase))


if __name__ == "__main__":
    main()

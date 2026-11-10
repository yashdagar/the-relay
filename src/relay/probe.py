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


PHASES = {"limits": limits}


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

import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field, fields, replace
from datetime import datetime

from relay.config import Config, ModelLimits
from relay.eval import grade
from relay.library import ROOT
from relay.pipeline import Pipeline
from relay.questions import QUESTIONS
from relay.records import result_row, write_run
from relay.standin import StandInConfig, running


@dataclass(frozen=True)
class Phase:
    duration_s: float
    rate_per_min: float


@dataclass(frozen=True)
class Scenario:
    name: str
    phases: tuple[Phase, ...]
    standin_overrides: dict = field(default_factory=dict)
    seed: int = 7

    @property
    def duration_s(self) -> float:
        return sum(p.duration_s for p in self.phases)

    def arrivals(self) -> list[float]:
        rng = random.Random(self.seed)
        offsets, start = [], 0.0
        for phase in self.phases:
            t = start
            while phase.rate_per_min > 0:
                t += rng.expovariate(phase.rate_per_min / 60)
                if t >= start + phase.duration_s:
                    break
                offsets.append(t)
            start += phase.duration_s
        return offsets

    def scaled(self, duration_s: float) -> "Scenario":
        factor = duration_s / self.duration_s
        phases = tuple(Phase(p.duration_s * factor, p.rate_per_min / factor) for p in self.phases)
        overrides = dict(self.standin_overrides)
        overrides["minute_s"] = overrides.get("minute_s", 60.0) * factor
        overrides["day_s"] = overrides.get("day_s", 86_400.0) * factor
        return replace(self, phases=phases, standin_overrides=overrides)

    def standin_config(self, **extra) -> StandInConfig:
        return StandInConfig(**{"content": "oracle", **self.standin_overrides, **extra})

    def configure(self, config: Config) -> Config:
        standin = self.standin_config()
        limits = ModelLimits(
            requests_per_minute=standin.rpm,
            tokens_per_minute=standin.tpm,
            requests_per_day=standin.rpd,
            tokens_per_day=standin.tpd,
        )
        return replace(
            config,
            limits={model: limits for model in config.limits},
            governor_minute_s=standin.minute_s,
            governor_day_s=standin.day_s,
        )


DEFAULTS = StandInConfig()
SCENARIOS = {
    s.name: s
    for s in (
        Scenario("steady", (Phase(1800, 2),)),
        Scenario("burst", (Phase(600, 1), Phase(30, 40), Phase(600, 1))),
        Scenario("overload", (Phase(1200, 6),)),
        Scenario(
            "rpm_halved", (Phase(1800, 2),), {"rpm": DEFAULTS.rpm // 2, "tpm": DEFAULTS.tpm // 2}
        ),
        Scenario("daily_exhausted", (Phase(1200, 2),), {"rpd": 15}),
    )
}


async def run(scenario: Scenario, config: Config, questions: list[dict], replies=None):
    config = scenario.configure(config)
    pipeline = Pipeline(config)
    arrivals = scenario.arrivals()
    rng = random.Random(scenario.seed)
    picks = [rng.choice(questions) for _ in arrivals]
    start = time.monotonic()

    async def one(i: int, offset: float, q: dict) -> dict:
        await asyncio.sleep(max(0.0, start + offset - time.monotonic()))
        started_at = time.time()
        reply = await pipeline.answer(q["question"])
        answer_grade = grade(q, reply)
        if replies is not None:
            replies.append(
                {"request_id": f"{scenario.name}-{i:04d}", "question_id": q["id"],
                 "grade": answer_grade, "reply": reply.to_dict()}
            )  # fmt: skip
        return result_row(
            request_id=f"{scenario.name}-{i:04d}",
            scenario=scenario.name,
            source=config.source,
            started_at=started_at,
            reply=reply,
            retrieval_setting=config.retrieval.name,
            answer_grade=answer_grade,
        )

    try:
        return list(
            await asyncio.gather(
                *(one(i, t, q) for i, (t, q) in enumerate(zip(arrivals, picks, strict=True)))
            )
        )
    finally:
        await pipeline.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive a load shape against the stand-in")
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--duration", type=float, help="compress the shape into this many seconds")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    if args.duration:
        scenario = scenario.scaled(args.duration)
    started = datetime.now()
    replies: list[dict] = []
    with running(scenario.standin_config()) as base_url:
        rows = asyncio.run(run(scenario, Config.standin(base_url), QUESTIONS, replies))
    out = ROOT / "results" / f"{started:%Y%m%d-%H%M}-load-{scenario.name}"
    standin = scenario.standin_config()
    info = {
        "started": started.isoformat(timespec="seconds"),
        "scenario": scenario.name,
        "phases": [vars(p) for p in scenario.phases],
        "requests": len(rows),
        "standin": {f.name: getattr(standin, f.name) for f in fields(standin)},
    }
    write_run(out, rows, sorted(replies, key=lambda r: r["request_id"]), info)
    print(out)


if __name__ == "__main__":
    main()

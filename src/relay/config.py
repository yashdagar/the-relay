import os
from dataclasses import dataclass, field, replace
from typing import Literal

from dotenv import load_dotenv

from relay.library import ROOT
from relay.retrieval import DEFAULT_SETTING, RetrievalSetting

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

UNDERSTAND_MODEL = "openai/gpt-oss-20b"
DRAFT_MODEL = "openai/gpt-oss-120b"
LEAN_DRAFT_MODEL = "openai/gpt-oss-20b"
CHECK_MODEL = "qwen/qwen3.8-27b"


@dataclass(frozen=True)
class ModelLimits:
    requests_per_minute: int = 30
    tokens_per_minute: int = 8000
    requests_per_day: int = 1000
    tokens_per_day: int = 200_000


GROQ_LIMITS = {
    "openai/gpt-oss-20b": ModelLimits(),
    "openai/gpt-oss-120b": ModelLimits(),
    "qwen/qwen3.8-27b": ModelLimits(),
}

MODEL_EXTRA = {
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
}


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    source: Literal["real", "stand-in"]
    understand_model: str = UNDERSTAND_MODEL
    draft_model: str = DRAFT_MODEL
    lean_draft_model: str = LEAN_DRAFT_MODEL
    check_model: str = CHECK_MODEL
    limits: dict[str, ModelLimits] = field(default_factory=lambda: dict(GROQ_LIMITS))
    retrieval: RetrievalSetting = DEFAULT_SETTING
    lean_top_k: int = 3
    refuse_below_score: float = 7.0
    max_extra_retrievals: int = 2
    deadline_ms: int = 15_000
    understand_timeout_ms: int = 4_000
    draft_timeout_ms: int = 8_000
    check_timeout_ms: int = 5_000
    understand_max_tokens: int = 300
    draft_max_tokens: int = 400
    check_max_tokens: int = 200
    max_retries: int = 2
    draft_reserve_ms: int = 1_700
    check_reserve_ms: int = 1_400
    governor_minute_s: float = 60.0
    governor_day_s: float = 86_400.0
    backoff_base_ms: int = 250
    max_drafts: int = 2
    max_model_calls: int = 5

    @classmethod
    def groq(cls, **overrides) -> "Config":
        load_dotenv(ROOT / ".env")
        return replace(
            cls(base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"], source="real"),
            **overrides,
        )

    @classmethod
    def standin(cls, base_url: str, **overrides) -> "Config":
        return replace(cls(base_url=base_url, api_key="stand-in", source="stand-in"), **overrides)

    def extra(self, model: str) -> dict:
        return MODEL_EXTRA.get(model, {})

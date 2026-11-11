from dataclasses import asdict, dataclass, field
from typing import Literal

from relay.library import Chunk

ReplyKind = Literal["answer", "refusal", "degraded"]
Status = Literal["ok", "rate_limited", "timeout", "error"]


@dataclass
class Reply:
    kind: ReplyKind
    text: str = ""
    reason: str | None = None
    citations: list[Chunk] = field(default_factory=list)
    retrieved: list[Chunk] = field(default_factory=list)
    status: Status = "ok"
    first_token_ms: float | None = None
    check_passed_ms: float | None = None
    total_ms: float | None = None
    retries: int = 0
    model_calls: int = 0
    drafts: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    queue_depth: int = 0
    steps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Reply":
        chunks = {key: [Chunk(**c) for c in data[key]] for key in ("citations", "retrieved")}
        return cls(**{**data, **chunks})

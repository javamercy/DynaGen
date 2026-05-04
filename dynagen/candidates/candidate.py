from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class CandidateStatus(StrEnum):
    PENDING = "pending"
    EVALUATED = "evaluated"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    VALID = "valid"
    ERROR = "error"


@dataclass
class Candidate:
    id: str
    generation: int
    strategy: str
    name: str = ""
    thought: str = ""
    code: str = ""
    parents: list[str] = field(default_factory=list)
    fitness: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    raw_response: str = ""
    status: CandidateStatus = CandidateStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        self.generation = int(self.generation)
        self.parents = list(self.parents)
        self.status = CandidateStatus(self.status)

    def to_dict(self, *, include_code: bool = True) -> dict[str, Any]:
        data = {
            "id": self.id,
            "generation": self.generation,
            "strategy": self.strategy,
            "name": self.name,
            "thought": self.thought,
            "parents": self.parents,
            "fitness": self.fitness,
            "metrics": self.metrics,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "status": self.status.value,
            "created_at": self.created_at,
        }

        if include_code:
            data["code"] = self.code
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, code: str | None = None) -> "Candidate":
        candidate_dict = dict(data)
        candidate_id = candidate_dict.pop("candidate_id", None)
        if "id" not in candidate_dict and candidate_id is not None:
            candidate_dict["id"] = candidate_id

        if code is not None:
            candidate_dict["code"] = code

        candidate_dict.setdefault("code", "")
        return cls(**candidate_dict)

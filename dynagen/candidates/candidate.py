from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

MINIMIZED_SCORE_NAMES = {"distance", "ttt"}


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
    distance: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    raw_response: str = ""
    error_details: str | None = None
    status: CandidateStatus = CandidateStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        self.generation = int(self.generation)
        self.parents = list(self.parents)
        self.status = CandidateStatus(self.status)
        if self._uses_distance():
            if self.distance is None:
                score_name = self.score_name
                self.distance = _first_float_or_none(
                    self.metrics.get(score_name),
                    self.metrics.get("ttt"),
                    self.metrics.get("distance"),
                    self.fitness,
                )
            self.fitness = None

    @property
    def score_name(self) -> str:
        return self._minimized_score_name() if self._uses_distance() else "fitness"

    @property
    def score_value(self) -> float | None:
        if self._uses_distance():
            score_name = self.score_name
            return _first_float_or_none(
                self.metrics.get(score_name),
                self.distance,
                self.metrics.get("ttt"),
                self.metrics.get("distance"),
            )
        return _float_or_none(self.fitness)

    def to_dict(self, *, include_code: bool = True) -> dict[str, Any]:
        data = {
            "id": self.id,
            "generation": self.generation,
            "strategy": self.strategy,
            "name": self.name,
            "thought": self.thought,
            "parents": self.parents,
            "metrics": self.metrics,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "error_details": self.error_details,
            "status": self.status.value,
            "created_at": self.created_at,
        }

        if self._uses_distance():
            data[self.score_name] = self.score_value
        else:
            data["fitness"] = self.fitness

        if include_code:
            data["code"] = self.code
        return data

    def _uses_distance(self) -> bool:
        if self.distance is not None:
            return True
        if not isinstance(self.metrics, dict):
            return False
        score_name = self.metrics.get("score_name")
        return (
            self.metrics.get("problem") == "tsp"
            or self.metrics.get("problem") == "dvrp"
            or score_name in MINIMIZED_SCORE_NAMES
            or "distance" in self.metrics
            or "ttt" in self.metrics
        )

    def _minimized_score_name(self) -> str:
        if isinstance(self.metrics, dict) and self.metrics.get("problem") == "dvrp":
            return "ttt"
        if isinstance(self.metrics, dict):
            score_name = self.metrics.get("score_name")
            if score_name in MINIMIZED_SCORE_NAMES:
                return str(score_name)
            if "ttt" in self.metrics:
                return "ttt"
        return "distance"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, code: str | None = None) -> "Candidate":
        candidate_dict = dict(data)
        candidate_id = candidate_dict.pop("candidate_id", None)
        if "id" not in candidate_dict and candidate_id is not None:
            candidate_dict["id"] = candidate_id

        ttt = candidate_dict.pop("ttt", None)
        if ttt is not None:
            candidate_dict.setdefault("distance", ttt)
            metrics = dict(candidate_dict.get("metrics") or {})
            metrics.setdefault("score_name", "ttt")
            metrics.setdefault("ttt", ttt)
            candidate_dict["metrics"] = metrics

        if code is not None:
            candidate_dict["code"] = code

        candidate_dict.setdefault("code", "")
        candidate_dict.setdefault("error_details", None)
        return cls(**candidate_dict)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _first_float_or_none(*values: object) -> float | None:
    for value in values:
        number = _float_or_none(value)
        if number is not None:
            return number
    return None

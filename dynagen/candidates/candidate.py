from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

MINIMIZED_SCORE_NAMES = {"distance", "gap", "ttt"}
MAXIMIZED_SCORE_NAMES = {"mean_aocc"}
NAMED_SCORE_NAMES = MINIMIZED_SCORE_NAMES | MAXIMIZED_SCORE_NAMES


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
        score_name = self._configured_score_name()
        if score_name in MINIMIZED_SCORE_NAMES:
            if self.distance is None:
                self.distance = _first_float_or_none(
                    self.metrics.get(score_name),
                    self.metrics.get("gap"),
                    self.metrics.get("ttt"),
                    self.metrics.get("distance"),
                    self.fitness,
                )
            self.fitness = None
        elif score_name in MAXIMIZED_SCORE_NAMES:
            if isinstance(self.metrics, dict):
                self.metrics.setdefault("score_name", score_name)
            self.distance = None
            self.fitness = None

    @property
    def score_name(self) -> str:
        return self._configured_score_name() or "fitness"

    @property
    def score_value(self) -> float | None:
        score_name = self.score_name
        if score_name in MINIMIZED_SCORE_NAMES:
            return _first_float_or_none(
                self.metrics.get(score_name),
                self.distance,
                self.metrics.get("gap"),
                self.metrics.get("ttt"),
                self.metrics.get("distance"),
            )
        if score_name in MAXIMIZED_SCORE_NAMES:
            return _first_float_or_none(self.metrics.get(score_name))
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

        if self.score_name != "fitness":
            data[self.score_name] = self.score_value
        else:
            data["fitness"] = self.fitness

        if include_code:
            data["code"] = self.code
        return data

    def _uses_distance(self) -> bool:
        return self.score_name in MINIMIZED_SCORE_NAMES

    def _configured_score_name(self) -> str | None:
        if not isinstance(self.metrics, dict):
            return "distance" if self.distance is not None else None
        score_name = self.metrics.get("score_name")
        if score_name in NAMED_SCORE_NAMES:
            return str(score_name)
        problem = self.metrics.get("problem")
        if problem == "bbob":
            return "mean_aocc"
        if problem == "dvrp":
            return "gap"
        if problem == "vrp":
            return "gap"
        if problem == "tsp":
            return "distance"
        if "gap" in self.metrics:
            return "gap"
        if "ttt" in self.metrics:
            return "ttt"
        if "distance" in self.metrics or self.distance is not None:
            return "distance"
        if "mean_aocc" in self.metrics:
            return "mean_aocc"
        return None

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

        gap = candidate_dict.pop("gap", None)
        if gap is not None:
            candidate_dict.setdefault("distance", gap)
            metrics = dict(candidate_dict.get("metrics") or {})
            metrics.setdefault("problem", "vrp")
            metrics.setdefault("score_name", "gap")
            metrics.setdefault("gap", gap)
            candidate_dict["metrics"] = metrics

        mean_aocc = candidate_dict.pop("mean_aocc", None)
        if mean_aocc is not None:
            metrics = dict(candidate_dict.get("metrics") or {})
            metrics.setdefault("problem", "bbob")
            metrics.setdefault("score_name", "mean_aocc")
            metrics.setdefault("mean_aocc", mean_aocc)
            candidate_dict["metrics"] = metrics
            candidate_dict["fitness"] = None

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

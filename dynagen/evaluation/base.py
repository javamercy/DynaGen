from dataclasses import dataclass
from typing import Any, Literal, Protocol

from dynagen.candidates.candidate import Candidate


EvaluationStatus = Literal["valid", "invalid", "timeout", "error"]


@dataclass(frozen=True)
class EvaluationResult:
    status: EvaluationStatus
    score: float | None
    metrics: dict[str, Any]
    error_feedback: str | None = None
    score_name: str = "fitness"

    @property
    def fitness(self) -> float | None:
        return self.score


class CandidateEvaluator(Protocol):
    def empty_metrics(self) -> dict[str, Any]:
        ...

    def evaluate_candidate(self, candidate: Candidate) -> EvaluationResult:
        ...

    def evaluate_code(self, code: str) -> EvaluationResult:
        ...

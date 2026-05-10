from dataclasses import dataclass
from typing import Any, Literal, Protocol

from dynagen.candidates.candidate import Candidate


EvaluationStatus = Literal["valid", "invalid", "timeout", "error"]


@dataclass(frozen=True)
class EvaluationResult:
    status: EvaluationStatus
    fitness: float | None
    metrics: dict[str, Any]
    error_feedback: str | None = None


class CandidateEvaluator(Protocol):
    def empty_metrics(self) -> dict[str, Any]:
        ...

    def evaluate_candidate(self, candidate: Candidate) -> EvaluationResult:
        ...

    def evaluate_code(self, code: str) -> EvaluationResult:
        ...

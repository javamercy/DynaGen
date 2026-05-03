import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
from dynagen.candidates.validation import validate_generated_code
from dynagen.domain.tsp_instance import TSPInstance
from dynagen.evaluation.metrics import aggregate_records, compute_gap
from dynagen.execution.runner import SolverRunStatus, run_solver


class EvaluationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class EvaluationResult:
    status: EvaluationStatus
    fitness: float | None
    metrics: dict[str, Any]
    error_feedback: str | None = None


class CandidateEvaluator:
    def __init__(
            self,
            instances: list[TSPInstance],
            *,
            seeds: tuple[int, ...] | list[int],
            budget: int,
            timeout_seconds: float,
            pool_name: str
    ) -> None:
        if not instances:
            raise ValueError("At least one instance is required for evaluation")
        if budget <= 0:
            raise ValueError("Budget must be a positive integer")
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be a positive number")
        if not pool_name:
            raise ValueError("Pool name cannot be empty")

        self.instances = tuple(instances)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.budget = int(budget)
        self.timeout_seconds = float(timeout_seconds)
        self.pool_name = pool_name

    def evaluate_candidate(self, candidate: Candidate) -> EvaluationResult:
        result = self.evaluate_code(candidate.code)
        candidate.status = CandidateStatus(result.status.value)  # TODO: fix here
        candidate.fitness = result.fitness
        candidate.metrics = result.metrics
        return result

    def evaluate_code(self, code: str) -> EvaluationResult:
        validation = validate_generated_code(code)
        if not validation.valid:
            metrics = self._with_context(aggregate_records([]))
            return EvaluationResult(EvaluationStatus.INVALID, math.inf, metrics)

        records: list[dict[str, Any]] = []

        for instance in self.instances:
            for seed in self.seeds:
                run = run_solver(
                    code,
                    instance,
                    seed=seed,
                    budget=self.budget,
                    timeout_seconds=self.timeout_seconds,
                )

                is_valid = run.status == SolverRunStatus.VALID
                gap = compute_gap(run.tour_length,
                                  instance.optimal_length) if is_valid and run.tour_length is not None else None
                records.append({
                    "instance": instance.name,
                    "pool": self.pool_name,
                    "dimension": instance.dimension,
                    "source": instance.metadata.get("source", "unknown"),
                    "seed": seed,
                    "status": run.status.value,
                    "tour_length": run.tour_length,
                    "reference_length": instance.optimal_length,
                    "reference_kind": "optimal",
                    "gap": gap,
                    "runtime_seconds": run.runtime_seconds,
                    "error": run.error,
                })
        metrics = self._with_context(aggregate_records(records))
        status = _candidate_status(metrics)
        fitness = metrics["mean_gap"] if status == EvaluationStatus.VALID else math.inf
        error_feedback = _error_feedback(records) if status != EvaluationStatus.VALID else None
        return EvaluationResult(status, fitness, metrics, error_feedback)

    def _with_context(self, metrics: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(metrics)
        metrics["pool"] = self.pool_name
        metrics["seeds"] = list(self.seeds)
        metrics["budget"] = self.budget
        return metrics


def _candidate_status(metrics: dict[str, Any]) -> EvaluationStatus:
    if metrics["timeout_count"]:
        return EvaluationStatus.TIMEOUT
    if metrics["runtime_error_count"]:
        return EvaluationStatus.ERROR
    if metrics["invalid_tour_count"]:
        return EvaluationStatus.INVALID
    if metrics["runs"] and metrics["valid_count"] == metrics["runs"]:
        return EvaluationStatus.VALID
    return EvaluationStatus.INVALID


def _error_feedback(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        if record["status"] != "valid":
            message = record.get("error") or record["status"]
            return f"{record['status']} on {record['instance']} seed {record['seed']}: {message}"
    return None

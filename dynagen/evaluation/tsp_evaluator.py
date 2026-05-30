import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
from dynagen.candidates.validation import validate_generated_code
from dynagen.domain.tsp_instance import TSPInstance
from dynagen.evaluation.base import EvaluationResult, EvaluationStatus
from dynagen.evaluation.tsp_metrics import aggregate_tsp_records, compute_gap
from dynagen.execution.tsp_runner import run_tsp_solver


class TSPCandidateEvaluator:
    def __init__(
            self,
            instances: list[TSPInstance],
            *,
            timeout_seconds: float | None,
            pool_name: str,
            timeout_penalty: float
    ) -> None:
        if not instances:
            raise ValueError("At least one instance is required for evaluation")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("Timeout must be a positive number")
        if timeout_penalty < 0:
            raise ValueError("Timeout penalty must be non-negative")
        if not pool_name:
            raise ValueError("Pool name cannot be empty")

        self.instances = tuple(instances)
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds is not None else None
        self.timeout_penalty = float(timeout_penalty)
        self.pool_name = pool_name

    def empty_metrics(self) -> dict[str, Any]:
        return self._with_context(aggregate_tsp_records([], timeout_penalty=self.timeout_penalty))

    def evaluate_candidate(self, candidate: Candidate) -> EvaluationResult:
        result = self.evaluate_code(candidate.code)
        candidate.status = CandidateStatus(result.status)
        candidate.distance = result.score
        candidate.metrics = dict(result.metrics)
        candidate.metrics["distance"] = result.score
        candidate.error_details = result.error_feedback
        return result

    def evaluate_code(self, code: str) -> EvaluationResult:
        validation = validate_generated_code(code)
        if not validation.valid:
            metrics = self.empty_metrics()
            metrics["distance"] = math.inf
            return EvaluationResult("invalid", math.inf, metrics, validation.error, score_name="distance")

        records = self._run_all_instances(code)
        metrics = self._with_context(aggregate_tsp_records(records, timeout_penalty=self.timeout_penalty))
        status = _candidate_status(metrics)
        distance = _candidate_distance(status, metrics, pool_name=self.pool_name)
        metrics["distance"] = distance
        error_feedback = _error_feedback(records) if status != "valid" else None
        return EvaluationResult(status, distance, metrics, error_feedback, score_name="distance")

    def _run_all_instances(self, code: str) -> list[dict[str, Any]]:
        tasks = list(self.instances)
        records: list[dict[str, Any]] = [None] * len(tasks)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 1)) as executor:
            future_to_index = {
                executor.submit(self._run_single_instance, code, instance): index
                for index, instance in enumerate(tasks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                records[index] = future.result()

        return records

    def _run_single_instance(self, code: str, instance: TSPInstance) -> dict[str, Any]:
        run = run_tsp_solver(
            code,
            instance,
            timeout_seconds=self.timeout_seconds,
        )
        scored = run.status == "valid" or run.partial
        gap = compute_gap(run.tour_length,
                          instance.optimal_length) if scored and run.tour_length is not None else None
        reference_kind = "optimal" if instance.optimal_length is not None else None
        return {
            "instance": instance.name,
            "pool": self.pool_name,
            "dimension": instance.dimension,
            "source": instance.metadata.get("source", "unknown"),
            "status": run.status,
            "tour_length": run.tour_length,
            "partial": run.partial,
            "reference_length": instance.optimal_length,
            "reference_kind": reference_kind,
            "gap": gap,
            "runtime_seconds": run.runtime_seconds,
            "timeout_limit_seconds": run.timeout_limit_seconds,
            "error": run.error,
        }

    def _with_context(self, metrics: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(metrics)
        metrics["problem"] = "tsp"
        metrics["score_name"] = "distance"
        metrics.setdefault("distance", math.inf)
        metrics["pool"] = self.pool_name
        metrics["timeout_seconds"] = self.timeout_seconds
        metrics["timeout_penalty"] = self.timeout_penalty
        return metrics


def _candidate_status(metrics: dict[str, Any]) -> EvaluationStatus:
    if metrics["timeout_count"]:
        return "timeout"
    if metrics["runtime_error_count"]:
        return "error"
    if metrics["invalid_tour_count"]:
        return "invalid"
    if metrics["runs"] and metrics["valid_count"] == metrics["runs"]:
        return "valid"
    return "invalid"


def _candidate_distance(status: EvaluationStatus, metrics: dict[str, Any], *, pool_name: str) -> float:
    if status == "valid":
        if pool_name == "search_instances":
            return metrics["mean_tour_length"] if metrics["mean_tour_length"] is not None else math.inf
        if metrics["mean_gap"] is not None:
            return metrics["mean_gap"]
        return metrics["mean_tour_length"] if metrics["mean_tour_length"] is not None else math.inf
    if status == "timeout":
        if pool_name == "search_instances" and metrics["mean_tour_length"] is not None:
            return metrics["mean_tour_length"]
        if metrics["penalized_mean_gap"] is not None:
            return metrics["penalized_mean_gap"]
        if metrics["mean_tour_length"] is not None:
            return metrics["mean_tour_length"]
        timeout_distance = metrics.get("timeout_distance")
        return float(timeout_distance) if timeout_distance is not None else math.inf
    return math.inf


def _error_feedback(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        if record["status"] != "valid":
            message = record.get("error") or record["status"]
            return f"{record['status']} on {record['instance']}: {message}"
    return None

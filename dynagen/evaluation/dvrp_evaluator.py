import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
from dynagen.candidates.validation import validate_dvrp_generated_code
from dynagen.domain.dvrp import DVRPInstance
from dynagen.evaluation.base import EvaluationResult, EvaluationStatus
from dynagen.evaluation.dvrp_metrics import aggregate_dvrp_records, compute_dvrp_gap
from dynagen.execution.dvrp_runner import run_dvrp_policy


class DVRPCandidateEvaluator:
    def __init__(
            self,
            instances: list[DVRPInstance],
            *,
            seeds: tuple[int, ...] | list[int],
            budget: int,
            timeout_seconds: float,
            pool_name: str,
            timeout_penalty: float = 10.0,
    ) -> None:
        if not instances:
            raise ValueError("At least one DVRP instance is required for evaluation")
        if budget <= 0:
            raise ValueError("Budget must be a positive integer")
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be a positive number")
        if timeout_penalty < 0:
            raise ValueError("Timeout penalty must be non-negative")
        if not pool_name:
            raise ValueError("Pool name cannot be empty")

        self.instances = tuple(instances)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.budget = int(budget)
        self.timeout_seconds = float(timeout_seconds)
        self.timeout_penalty = float(timeout_penalty)
        self.pool_name = pool_name

    def empty_metrics(self) -> dict[str, Any]:
        return self._with_context(aggregate_dvrp_records([], timeout_penalty=self.timeout_penalty))

    def evaluate_candidate(self, candidate: Candidate) -> EvaluationResult:
        result = self.evaluate_code(candidate.code)
        candidate.status = CandidateStatus(result.status)
        candidate.distance = result.score
        candidate.fitness = None
        candidate.metrics = dict(result.metrics)
        candidate.metrics["distance"] = result.score
        candidate.error_details = result.error_feedback
        return result

    def evaluate_code(self, code: str) -> EvaluationResult:
        validation = validate_dvrp_generated_code(code)
        if not validation.valid:
            metrics = self.empty_metrics()
            metrics["distance"] = math.inf
            return EvaluationResult("invalid", math.inf, metrics, validation.error, score_name="distance")

        records = self._run_all_instances(code)
        metrics = self._with_context(aggregate_dvrp_records(records, timeout_penalty=self.timeout_penalty))
        status = _candidate_status(metrics)
        distance = _candidate_distance(status, metrics)
        metrics["distance"] = distance
        error_feedback = _error_feedback(records) if status != "valid" else None
        return EvaluationResult(status, distance, metrics, error_feedback, score_name="distance")

    def _run_all_instances(self, code: str) -> list[dict[str, Any]]:
        tasks = [
            (instance, seed)
            for instance in self.instances
            for seed in self.seeds
        ]
        records: list[dict[str, Any]] = [None] * len(tasks)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
            future_to_index = {
                executor.submit(self._run_single_instance, code, instance, seed): index
                for index, (instance, seed) in enumerate(tasks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                records[index] = future.result()

        return records

    def _run_single_instance(self, code: str, instance: DVRPInstance, seed: int) -> dict[str, Any]:
        run = run_dvrp_policy(
            code,
            instance,
            seed=seed,
            budget=self.budget,
            timeout_seconds=self.timeout_seconds,
        )
        scored = run.status == "valid" and run.makespan is not None
        gap = compute_dvrp_gap(run.makespan, instance.reference_makespan) if scored else None
        return {
            "instance": instance.name,
            "pool": self.pool_name,
            "dimension": instance.dimension,
            "customer_count": instance.customer_count,
            "truck_count": instance.truck_count,
            "source": instance.metadata.get("source", "unknown"),
            "seed": seed,
            "status": run.status,
            "makespan": run.makespan,
            "reference_makespan": instance.reference_makespan,
            "reference_kind": "ortools_static_full_future" if instance.reference_makespan is not None else None,
            "gap": gap,
            "decisions": run.decisions,
            "waits": run.waits,
            "completed_count": run.completed_count,
            "runtime_seconds": run.runtime_seconds,
            "timeout_limit_seconds": run.timeout_limit_seconds,
            "error": run.error,
        }

    def _with_context(self, metrics: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(metrics)
        metrics["problem"] = "dvrp"
        metrics["score_name"] = "distance"
        metrics.setdefault("distance", math.inf)
        metrics["pool"] = self.pool_name
        metrics["seeds"] = list(self.seeds)
        metrics["budget"] = self.budget
        metrics["timeout_seconds"] = self.timeout_seconds
        metrics["timeout_penalty"] = self.timeout_penalty
        return metrics


def _candidate_status(metrics: dict[str, Any]) -> EvaluationStatus:
    if metrics["timeout_count"]:
        return "timeout"
    if metrics["runtime_error_count"]:
        return "error"
    if metrics["invalid_count"]:
        return "invalid"
    if metrics["runs"] and metrics["valid_count"] == metrics["runs"]:
        return "valid"
    return "invalid"


def _candidate_distance(status: EvaluationStatus, metrics: dict[str, Any]) -> float:
    if status == "valid":
        if metrics["mean_gap"] is not None:
            return float(metrics["mean_gap"])
        if metrics["mean_makespan"] is not None:
            return float(metrics["mean_makespan"])
        return math.inf
    if status == "timeout":
        timeout_distance = metrics.get("timeout_distance")
        return float(timeout_distance) if timeout_distance is not None else math.inf
    return math.inf


def _error_feedback(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        if record["status"] != "valid":
            message = record.get("error") or record["status"]
            return f"{record['status']} on {record['instance']} seed {record['seed']}: {message}"
    return None

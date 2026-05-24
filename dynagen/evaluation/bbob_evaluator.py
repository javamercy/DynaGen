import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
from dynagen.candidates.validation import validate_bbob_generated_code
from dynagen.domain.bbob import BBOBInstance
from dynagen.evaluation.base import EvaluationResult, EvaluationStatus
from dynagen.evaluation.bbob_metrics import aggregate_bbob_records, compute_aocc
from dynagen.execution.bbob_runner import run_bbob_optimizer


class BBOBCandidateEvaluator:
    def __init__(
            self,
            instances: list[BBOBInstance],
            *,
            seeds: tuple[int, ...] | list[int],
            budget: int,
            timeout_seconds: float = 600,
            pool_name: str,
            timeout_penalty: float = 0.0,
            aocc_lower_bound: float = 1e-8,
            aocc_upper_bound: float = 1e2,
    ) -> None:
        if not instances:
            raise ValueError("At least one BBOB instance is required for evaluation")
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
        self.aocc_lower_bound = float(aocc_lower_bound)
        self.aocc_upper_bound = float(aocc_upper_bound)

    def empty_metrics(self) -> dict[str, Any]:
        return self._with_context(aggregate_bbob_records([], timeout_penalty=self.timeout_penalty))

    def evaluate_candidate(self, candidate: Candidate) -> EvaluationResult:
        result = self.evaluate_code(candidate.code)
        candidate.status = CandidateStatus(result.status)
        candidate.fitness = None
        candidate.distance = None
        candidate.metrics = dict(result.metrics)
        candidate.metrics["score_name"] = "mean_aocc"
        candidate.metrics["mean_aocc"] = result.score
        candidate.error_details = result.error_feedback
        return result

    def evaluate_code(self, code: str) -> EvaluationResult:
        validation = validate_bbob_generated_code(code)
        if not validation.valid:
            metrics = self.empty_metrics()
            metrics["mean_aocc"] = 0.0
            return EvaluationResult("invalid", 0.0, metrics, validation.error, score_name="mean_aocc")

        records = self._run_all_instances(code)
        metrics = self._with_context(aggregate_bbob_records(records, timeout_penalty=self.timeout_penalty))
        status = _candidate_status(metrics)
        mean_aocc = _candidate_mean_aocc_score(status, metrics)
        metrics["mean_aocc"] = mean_aocc
        error_feedback = _error_feedback(records) if status != "valid" else None
        return EvaluationResult(status, mean_aocc, metrics, error_feedback, score_name="mean_aocc")

    def _run_all_instances(self, code: str) -> list[dict[str, Any]]:
        tasks = [
            (instance, seed)
            for instance in self.instances
            for seed in self.seeds
        ]
        records: list[dict[str, Any]] = [None] * len(tasks)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 1)) as executor:
            future_to_index = {
                executor.submit(self._run_single_instance, code, instance, seed): index
                for index, (instance, seed) in enumerate(tasks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                records[index] = future.result()

        return records

    def _run_single_instance(self, code: str, instance: BBOBInstance, seed: int) -> dict[str, Any]:
        run = run_bbob_optimizer(
            code,
            instance,
            seed=seed,
            budget=self.budget,
            timeout_seconds=self.timeout_seconds,
        )
        history = run.history or ([run.best_value] if run.best_value is not None else [])
        aocc = compute_aocc(
            history,
            optimum=instance.optimum_value,
            budget=self.budget,
            lower_bound=self.aocc_lower_bound,
            upper_bound=self.aocc_upper_bound,
        ) if history else None
        final_error = None
        if run.best_value is not None and math.isfinite(run.best_value):
            final_error = max(0.0, float(run.best_value) - instance.optimum_value)
        return {
            "instance": instance.name,
            "pool": self.pool_name,
            "function_id": instance.function_id,
            "function_name": instance.name.split("_i", 1)[0],
            "group": instance.group,
            "bbob_instance_id": instance.instance_id,
            "dimension": instance.dimension,
            "seed": seed,
            "status": run.status,
            "best_value": run.best_value,
            "final_error": final_error,
            "aocc": aocc,
            "evaluations": run.evaluations,
            "partial": run.partial,
            "runtime_seconds": run.runtime_seconds,
            "timeout_limit_seconds": run.timeout_limit_seconds,
            "error": run.error,
        }

    def _with_context(self, metrics: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(metrics)
        metrics["problem"] = "bbob"
        metrics["score_name"] = "mean_aocc"
        metrics["pool"] = self.pool_name
        metrics["seeds"] = list(self.seeds)
        metrics["budget"] = self.budget
        metrics["timeout_seconds"] = self.timeout_seconds
        metrics["timeout_penalty"] = self.timeout_penalty
        metrics["aocc_lower_bound"] = self.aocc_lower_bound
        metrics["aocc_upper_bound"] = self.aocc_upper_bound
        return metrics


def _candidate_status(metrics: dict[str, Any]) -> EvaluationStatus:
    if metrics["runtime_error_count"]:
        return "error"
    if metrics["timeout_count"]:
        return "timeout"
    if metrics["invalid_count"]:
        return "invalid"
    if metrics["runs"] and metrics["valid_count"] == metrics["runs"]:
        return "valid"
    return "invalid"


def _candidate_mean_aocc_score(status: EvaluationStatus, metrics: dict[str, Any]) -> float:
    if status == "valid":
        return _finite_or_zero(metrics.get("mean_aocc"))
    if status == "timeout":
        return _finite_or_zero(metrics.get("timeout_mean_aocc"))
    return 0.0


def _finite_or_zero(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _error_feedback(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        if record["status"] != "valid":
            message = record.get("error") or record["status"]
            return f"{record['status']} on {record['instance']} seed {record['seed']}: {message}"
    return None

from dataclasses import dataclass

import traceback
from typing import Literal

from dynagen.domain.tsp_instance import TSPInstance
from dynagen.execution.timeouts import execute_solver_code


SolverRunStatus = Literal["valid", "invalid", "timeout", "error"]


@dataclass(frozen=True)
class SolverRunResult:
    status: SolverRunStatus
    tour: list[int] | None
    tour_length: float | None
    runtime_seconds: float
    error: str | None = None
    partial: bool = False


def run_solver(
        code: str,
        instance: TSPInstance,
        *,
        seed: int,
        budget: int,
        timeout_seconds: float,
) -> SolverRunResult:
    execution = execute_solver_code(
        code,
        instance.distance_matrix,
        seed=seed,
        budget=budget,
        timeout_seconds=timeout_seconds,
    )

    if execution.status == "timeout":
        if execution.reported_value is not None:
            try:
                tour, length = _validated_tour(instance, execution.reported_value)
                return SolverRunResult("timeout", tour, length, execution.runtime_seconds, execution.error, partial=True)
            except Exception as exc:
                error = f"{execution.error}; reported best tour invalid: {exc}"
                return SolverRunResult("timeout", None, None, execution.runtime_seconds, error)
        return SolverRunResult("timeout", None, None, execution.runtime_seconds, execution.error)

    if execution.status != "ok":
        return SolverRunResult("error", None, None, execution.runtime_seconds, execution.error)

    try:
        tour, length = _validated_tour(instance, execution.value)
    except Exception:
        return SolverRunResult(
            "invalid",
            None,
            None,
            execution.runtime_seconds,
            error=traceback.format_exc()
        )
    return SolverRunResult("valid", tour, length, execution.runtime_seconds)


def _validated_tour(instance: TSPInstance, value: object) -> tuple[list[int], float]:
    tour = instance.validate_tour(value).astype(int).tolist()
    return tour, instance.tour_length(tour)

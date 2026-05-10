from dataclasses import dataclass

from typing import Literal

from dynagen.domain.tsp_instance import TSPInstance
from dynagen.execution.tsp_timeouts import execute_tsp_solver_code


TSPSolverRunStatus = Literal["valid", "invalid", "timeout", "error"]


@dataclass(frozen=True)
class TSPSolverRunResult:
    status: TSPSolverRunStatus
    tour: list[int] | None
    tour_length: float | None
    runtime_seconds: float
    error: str | None = None
    partial: bool = False


def run_tsp_solver(
        code: str,
        instance: TSPInstance,
        *,
        seed: int,
        budget: int,
        timeout_seconds: float,
) -> TSPSolverRunResult:
    execution = execute_tsp_solver_code(
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
                return TSPSolverRunResult("timeout", tour, length, execution.runtime_seconds, execution.error, partial=True)
            except Exception as exc:
                error = f"{execution.error}; reported best tour invalid: {exc}"
                return TSPSolverRunResult("timeout", None, None, execution.runtime_seconds, error)
        return TSPSolverRunResult("timeout", None, None, execution.runtime_seconds, execution.error)

    if execution.status != "ok":
        return TSPSolverRunResult("error", None, None, execution.runtime_seconds, execution.error)

    try:
        tour, length = _validated_tour(instance, execution.value)
    except Exception as exc:
        return TSPSolverRunResult(
            "invalid",
            None,
            None,
            execution.runtime_seconds,
            error=_short_error_message(exc)
        )
    return TSPSolverRunResult("valid", tour, length, execution.runtime_seconds)


def _validated_tour(instance: TSPInstance, value: object) -> tuple[list[int], float]:
    tour = instance.validate_tour(value).astype(int).tolist()
    return tour, instance.tour_length(tour)


def _short_error_message(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__

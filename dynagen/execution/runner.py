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
        return SolverRunResult("timeout", None, None, execution.runtime_seconds, execution.error)

    if execution.status != "ok":
        return SolverRunResult("error", None, None, execution.runtime_seconds, execution.error)

    try:
        tour = instance.validate_tour(execution.value).astype(int).tolist()
        length = instance.tour_length(tour)
    except Exception:
        return SolverRunResult(
            "invalid",
            None,
            None,
            execution.runtime_seconds,
            error=traceback.format_exc()
        )
    return SolverRunResult("valid", tour, length, execution.runtime_seconds)

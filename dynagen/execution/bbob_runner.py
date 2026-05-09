import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from dynagen.domain.bbob import BBOBInstance

BBOBExecutionStatus = Literal["ok", "timeout", "error"]
BBOBRunStatus = Literal["valid", "invalid", "timeout", "error"]


@dataclass(frozen=True)
class BBOBExecutionResult:
    status: BBOBExecutionStatus
    best_value: float | None = None
    best_x: list[float] | None = None
    history: list[float] = field(default_factory=list)
    evaluations: int = 0
    runtime_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class BBOBRunResult:
    status: BBOBRunStatus
    best_value: float | None
    best_x: list[float] | None
    history: list[float]
    evaluations: int
    runtime_seconds: float
    error: str | None = None
    partial: bool = False


def run_bbob_optimizer(
        code: str,
        instance: BBOBInstance,
        *,
        seed: int,
        budget: int,
        timeout_seconds: float,
) -> BBOBRunResult:
    execution = execute_bbob_code(
        code,
        instance,
        seed=seed,
        budget=budget,
        timeout_seconds=timeout_seconds,
    )
    if execution.status == "timeout":
        if execution.best_value is not None and np.isfinite(execution.best_value):
            return BBOBRunResult(
                "timeout",
                execution.best_value,
                execution.best_x,
                execution.history,
                execution.evaluations,
                execution.runtime_seconds,
                execution.error,
                partial=True,
            )
        return BBOBRunResult(
            "timeout",
            None,
            None,
            [],
            execution.evaluations,
            execution.runtime_seconds,
            execution.error,
        )
    if execution.status != "ok":
        return BBOBRunResult(
            "error",
            execution.best_value,
            execution.best_x,
            execution.history,
            execution.evaluations,
            execution.runtime_seconds,
            execution.error,
        )
    if execution.best_value is None or execution.best_x is None or not np.isfinite(execution.best_value):
        return BBOBRunResult(
            "invalid",
            None,
            None,
            execution.history,
            execution.evaluations,
            execution.runtime_seconds,
            "Optimizer did not evaluate and return a finite incumbent",
        )
    if len(execution.best_x) != instance.dimension:
        return BBOBRunResult(
            "invalid",
            None,
            None,
            execution.history,
            execution.evaluations,
            execution.runtime_seconds,
            f"Returned incumbent dimension {len(execution.best_x)} does not match {instance.dimension}",
        )
    return BBOBRunResult(
        "valid",
        execution.best_value,
        execution.best_x,
        execution.history,
        execution.evaluations,
        execution.runtime_seconds,
    )


def execute_bbob_code(
        code: str,
        instance: BBOBInstance,
        *,
        seed: int,
        budget: int,
        timeout_seconds: float,
) -> BBOBExecutionResult:
    context = _multiprocessing_context()
    best_x = context.Array("d", instance.dimension, lock=False)
    best_value = context.Value("d", float("inf"), lock=False)
    has_best = context.Value("i", 0, lock=False)
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_worker,
        args=(code, instance, int(seed), int(budget), result_queue, best_x, best_value, has_best),
    )
    start = time.perf_counter()
    process.start()
    process.join(timeout_seconds)
    runtime = time.perf_counter() - start
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        reported_value, reported_x = _reported_best(best_x, best_value, has_best)
        return BBOBExecutionResult(
            "timeout",
            best_value=reported_value,
            best_x=reported_x,
            runtime_seconds=runtime,
            error="Optimizer timed out",
        )
    try:
        status, value, point, history, evaluations, child_runtime, error = result_queue.get_nowait()
    except queue.Empty:
        reported_value, reported_x = _reported_best(best_x, best_value, has_best)
        if process.exitcode == 0:
            return BBOBExecutionResult(
                "error",
                best_value=reported_value,
                best_x=reported_x,
                runtime_seconds=runtime,
                error="Optimizer exited without returning a result",
            )
        return BBOBExecutionResult(
            "error",
            best_value=reported_value,
            best_x=reported_x,
            runtime_seconds=runtime,
            error=f"Optimizer process exited with code {process.exitcode}",
        )
    return BBOBExecutionResult(
        status,
        best_value=value,
        best_x=point,
        history=history or [],
        evaluations=int(evaluations or 0),
        runtime_seconds=child_runtime,
        error=error,
    )


def _worker(code: str, instance: BBOBInstance, seed: int, budget: int, result_queue, best_x, best_value,
            has_best) -> None:
    from dynagen.domain.bbob import BudgetedBBOBObjective
    from dynagen.execution.sandbox import load_bbob_optimizer

    def report_best(value: object, x: object) -> None:
        try:
            numeric_value = float(value)
            point = np.asarray(x, dtype=float).reshape(-1)
            if point.size != instance.dimension or not np.isfinite(numeric_value):
                return None
            if has_best.value and numeric_value >= best_value.value:
                return None
            for index, coordinate in enumerate(point):
                best_x[index] = float(coordinate)
            best_value.value = numeric_value
            has_best.value = 1
        except Exception:
            return None
        return None

    objective = None
    start = time.perf_counter()
    try:
        optimizer_cls = load_bbob_optimizer(code, best_value_reporter=report_best)
        objective = BudgetedBBOBObjective(instance, budget=budget, on_improvement=report_best)
        optimizer = optimizer_cls(budget=int(budget), dim=int(instance.dimension), seed=int(seed))
        optimizer(objective)
        runtime = time.perf_counter() - start
        if objective.best_x is None or not np.isfinite(objective.best_value):
            raise ValueError("Optimizer did not evaluate the objective")
        report_best(objective.best_value, objective.best_x)
        result_queue.put((
            "ok",
            float(objective.best_value),
            objective.best_x.astype(float).tolist(),
            [float(item) for item in objective.history],
            int(objective.evaluations),
            runtime,
            None,
        ))
    except Exception as exc:
        runtime = time.perf_counter() - start
        history = [] if objective is None else [float(item) for item in objective.history]
        evaluations = 0 if objective is None else int(objective.evaluations)
        value, point = _reported_best(best_x, best_value, has_best)
        result_queue.put(("error", value, point, history, evaluations, runtime, _short_error_message(exc)))


def _reported_best(best_x, best_value, has_best) -> tuple[float | None, list[float] | None]:
    if not has_best.value:
        return None, None
    return float(best_value.value), [float(item) for item in best_x]


def _short_error_message(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _multiprocessing_context():
    methods = mp.get_all_start_methods()
    if "spawn" in methods:
        return mp.get_context("spawn")
    if "forkserver" in methods:
        return mp.get_context("forkserver")
    return mp.get_context()

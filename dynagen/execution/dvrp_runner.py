import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from dynagen.domain.dvrp import DVRPInstance, DVRPSimulationResult


DVRPExecutionStatus = Literal["ok", "invalid", "timeout", "error"]
DVRPRunStatus = Literal["valid", "invalid", "timeout", "error"]


@dataclass(frozen=True)
class DVRPExecutionResult:
    status: DVRPExecutionStatus
    simulation: dict | None = None
    runtime_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class DVRPRunResult:
    status: DVRPRunStatus
    makespan: float | None
    routes: list[list[int]]
    decisions: int
    waits: int
    completed_count: int
    runtime_seconds: float
    error: str | None = None


def run_dvrp_policy(
        code: str,
        instance: DVRPInstance,
        *,
        seed: int,
        budget: int,
        timeout_seconds: float,
) -> DVRPRunResult:
    execution = execute_dvrp_code(
        code,
        instance,
        seed=seed,
        budget=budget,
        timeout_seconds=timeout_seconds,
    )
    if execution.status == "timeout":
        return DVRPRunResult("timeout", None, [], 0, 0, 0, execution.runtime_seconds, execution.error)
    if execution.status == "invalid":
        return DVRPRunResult("invalid", None, [], 0, 0, 0, execution.runtime_seconds, execution.error)
    if execution.status != "ok":
        return DVRPRunResult("error", None, [], 0, 0, 0, execution.runtime_seconds, execution.error)

    try:
        simulation = DVRPSimulationResult.from_dict(execution.simulation or {})
    except Exception as exc:
        return DVRPRunResult("invalid", None, [], 0, 0, 0, execution.runtime_seconds, _short_error_message(exc))

    if not np.isfinite(simulation.makespan) or simulation.makespan <= 0:
        return DVRPRunResult("invalid", None, [], simulation.decisions, simulation.waits, simulation.completed_count,
                             execution.runtime_seconds, "Policy produced an invalid makespan")
    return DVRPRunResult(
        "valid",
        simulation.makespan,
        simulation.routes,
        simulation.decisions,
        simulation.waits,
        simulation.completed_count,
        execution.runtime_seconds,
    )


def execute_dvrp_code(
        code: str,
        instance: DVRPInstance,
        *,
        seed: int,
        budget: int,
        timeout_seconds: float,
) -> DVRPExecutionResult:
    context = _multiprocessing_context()
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_worker,
        args=(code, instance, int(seed), int(budget), result_queue),
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
        return DVRPExecutionResult("timeout", runtime_seconds=runtime, error="DVRP policy timed out")
    try:
        status, simulation, child_runtime, error = result_queue.get_nowait()
    except queue.Empty:
        if process.exitcode == 0:
            return DVRPExecutionResult(
                "error",
                runtime_seconds=runtime,
                error="DVRP policy exited without returning a result",
            )
        return DVRPExecutionResult(
            "error",
            runtime_seconds=runtime,
            error=f"DVRP policy process exited with code {process.exitcode}",
        )
    return DVRPExecutionResult(status, simulation=simulation, runtime_seconds=child_runtime, error=error)


def _worker(code: str, instance: DVRPInstance, seed: int, budget: int, result_queue) -> None:
    from dynagen.domain.dvrp import DVRPSimulationError, simulate_dvrp_policy
    from dynagen.execution.sandbox import load_dvrp_policy

    start = time.perf_counter()
    try:
        policy = load_dvrp_policy(code)
        simulation = simulate_dvrp_policy(instance, policy, seed=seed, budget=budget)
        runtime = time.perf_counter() - start
        result_queue.put(("ok", simulation.to_dict(), runtime, None))
    except DVRPSimulationError as exc:
        runtime = time.perf_counter() - start
        result_queue.put(("invalid", None, runtime, _short_error_message(exc)))
    except Exception as exc:
        runtime = time.perf_counter() - start
        result_queue.put(("error", None, runtime, _short_error_message(exc)))


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

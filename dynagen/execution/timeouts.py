import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


SolverExecutionStatus = Literal["ok", "timeout", "error"]


@dataclass(frozen=True)
class SolverExecutionResult:
    status: SolverExecutionStatus
    value: Any = None
    runtime_seconds: float = 0.0
    error: str | None = None


def execute_solver_code(
        code: str,
        distance_matrix: np.ndarray,
        *,
        seed: int,
        budget: int,
        timeout_seconds: float,
) -> SolverExecutionResult:
    context = _multiprocessing_context()
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_worker,
                              args=(code, np.asarray(distance_matrix, dtype=float), seed, budget, result_queue))
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
        return SolverExecutionResult("timeout", runtime_seconds=runtime, error="Solver timed out")
    try:
        status, value, child_runtime, error = result_queue.get_nowait()
    except queue.Empty:
        if process.exitcode == 0:
            return SolverExecutionResult("error", runtime_seconds=runtime,
                                         error="Solver exited without returning a result")
        return SolverExecutionResult("error", runtime_seconds=runtime,
                                     error=f"Solver process exited with code {process.exitcode}")
    return SolverExecutionResult(status, value=value, runtime_seconds=child_runtime, error=error)


def _worker(code: str, distance_matrix: np.ndarray, seed: int, budget: int, result_queue) -> None:
    from dynagen.execution.sandbox import load_tsp_solver

    start = time.perf_counter()
    try:
        tsp_solver = load_tsp_solver(code)
        tour = tsp_solver(distance_matrix.copy(), int(seed), int(budget))
        runtime = time.perf_counter() - start
        result_queue.put(("ok", np.asarray(tour).tolist(), runtime, None))
    except Exception:
        runtime = time.perf_counter() - start
        result_queue.put(("error", None, runtime, traceback.format_exc(limit=20)))


def _multiprocessing_context():
    methods = mp.get_all_start_methods()
    if "spawn" in methods:
        return mp.get_context("spawn")
    if "forkserver" in methods:
        return mp.get_context("forkserver")
    return mp.get_context()

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
    reported_value: Any = None
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
    distance_matrix_arr = np.asarray(distance_matrix, dtype=float)
    dimension = int(distance_matrix_arr.shape[0])
    best_tour_a = context.Array("d", dimension, lock=False)
    best_tour_b = context.Array("d", dimension, lock=False)
    active_tour_index = context.Value("i", -1, lock=False)
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_worker,
                              args=(code, distance_matrix_arr, seed, budget, result_queue, best_tour_a, best_tour_b,
                                    active_tour_index))
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
        return SolverExecutionResult("timeout", reported_value=_reported_tour(best_tour_a, best_tour_b,
                                                                              active_tour_index),
                                     runtime_seconds=runtime, error="Solver timed out")
    try:
        status, value, child_runtime, error = result_queue.get_nowait()
    except queue.Empty:
        if process.exitcode == 0:
            return SolverExecutionResult("error", reported_value=_reported_tour(best_tour_a, best_tour_b,
                                                                                active_tour_index),
                                         runtime_seconds=runtime,
                                         error="Solver exited without returning a result")
        return SolverExecutionResult("error", runtime_seconds=runtime,
                                     reported_value=_reported_tour(best_tour_a, best_tour_b, active_tour_index),
                                     error=f"Solver process exited with code {process.exitcode}")
    return SolverExecutionResult(status, value=value,
                                 reported_value=_reported_tour(best_tour_a, best_tour_b, active_tour_index),
                                 runtime_seconds=child_runtime, error=error)


def _worker(code: str, distance_matrix: np.ndarray, seed: int, budget: int, result_queue, best_tour_a, best_tour_b,
            active_tour_index) -> None:
    from dynagen.execution.sandbox import load_tsp_solver

    def report_best_tour(tour: object) -> None:
        try:
            tour_arr = np.asarray(tour, dtype=float).reshape(-1)
            if tour_arr.size != distance_matrix.shape[0]:
                return None
            write_index = 1 if active_tour_index.value == 0 else 0
            target = best_tour_b if write_index == 1 else best_tour_a
            for index, node in enumerate(tour_arr):
                target[index] = float(node)
            active_tour_index.value = write_index
        except Exception:
            return None

    start = time.perf_counter()
    try:
        tsp_solver = load_tsp_solver(code, best_tour_reporter=report_best_tour)
        tour = tsp_solver(distance_matrix.copy(), int(seed), int(budget))
        runtime = time.perf_counter() - start
        result_queue.put(("ok", np.asarray(tour).tolist(), runtime, None))
    except Exception:
        runtime = time.perf_counter() - start
        result_queue.put(("error", None, runtime, traceback.format_exc(limit=20)))


def _reported_tour(best_tour_a, best_tour_b, active_tour_index) -> Any | None:
    if active_tour_index.value == 0:
        return list(best_tour_a)
    if active_tour_index.value == 1:
        return list(best_tour_b)
    return None


def _multiprocessing_context():
    methods = mp.get_all_start_methods()
    if "spawn" in methods:
        return mp.get_context("spawn")
    if "forkserver" in methods:
        return mp.get_context("forkserver")
    return mp.get_context()

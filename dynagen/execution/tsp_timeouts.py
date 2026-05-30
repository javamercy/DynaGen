import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


TSPSolverExecutionStatus = Literal["ok", "timeout", "error"]


@dataclass(frozen=True)
class TSPSolverExecutionResult:
    status: TSPSolverExecutionStatus
    value: Any = None
    reported_value: Any = None
    runtime_seconds: float = 0.0
    error: str | None = None
    timeout_limit_seconds: float | None = None


def execute_tsp_solver_code(
        code: str,
        distance_matrix: np.ndarray,
        *,
        timeout_seconds: float | None,
) -> TSPSolverExecutionResult:
    context = _multiprocessing_context()
    distance_matrix_arr = np.asarray(distance_matrix, dtype=float)
    dimension = int(distance_matrix_arr.shape[0])
    best_tour_a = context.Array("d", dimension, lock=False)
    best_tour_b = context.Array("d", dimension, lock=False)
    active_tour_index = context.Value("i", -1, lock=False)
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_worker,
                              args=(code, distance_matrix_arr, result_queue, best_tour_a, best_tour_b,
                                    active_tour_index))
    timeout_limit = _normalize_timeout_seconds(timeout_seconds)
    start = time.perf_counter()
    process.start()
    payload = _get_worker_result_until_deadline(result_queue, process, start=start, timeout_seconds=timeout_limit)
    runtime = time.perf_counter() - start
    if payload is not None:
        process.join(1.0)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive():
                process.kill()
                process.join()
        return _execution_result_from_payload(
            payload,
            reported_value=_reported_tour(best_tour_a, best_tour_b, active_tour_index),
            timeout_limit_seconds=timeout_limit,
        )
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return TSPSolverExecutionResult("timeout", reported_value=_reported_tour(best_tour_a, best_tour_b,
                                                                                   active_tour_index),
                                          runtime_seconds=runtime,
                                          error=(
                                              f"Solver timed out after {runtime:.6g}s (timeout_seconds={timeout_limit:.6g})"
                                              if timeout_limit is not None else
                                              f"Solver timed out after {runtime:.6g}s"
                                          ),
                                          timeout_limit_seconds=timeout_limit)
    try:
        status, value, child_runtime, error = result_queue.get_nowait()
    except queue.Empty:
        if process.exitcode == 0:
            return TSPSolverExecutionResult("error", reported_value=_reported_tour(best_tour_a, best_tour_b,
                                                                                   active_tour_index),
                                            runtime_seconds=runtime,
                                            error="Solver exited without returning a result",
                                            timeout_limit_seconds=timeout_limit)
        return TSPSolverExecutionResult("error", runtime_seconds=runtime,
                                         reported_value=_reported_tour(best_tour_a, best_tour_b, active_tour_index),
                                         error=f"Solver process exited with code {process.exitcode}",
                                         timeout_limit_seconds=timeout_limit)
    return TSPSolverExecutionResult(status, value=value,
                                     reported_value=_reported_tour(best_tour_a, best_tour_b, active_tour_index),
                                     runtime_seconds=child_runtime, error=error,
                                     timeout_limit_seconds=timeout_limit)


def _get_worker_result_until_deadline(result_queue, process, *, start: float, timeout_seconds: float | None):
    if timeout_seconds is None:
        while True:
            try:
                return result_queue.get(timeout=0.05)
            except queue.Empty:
                if not process.is_alive():
                    return None

    deadline = start + float(timeout_seconds)
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return None
        try:
            return result_queue.get(timeout=min(0.05, remaining))
        except queue.Empty:
            if not process.is_alive():
                return None


def _normalize_timeout_seconds(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    timeout_limit = float(timeout_seconds)
    if timeout_limit <= 0:
        raise ValueError("timeout_seconds must be positive")
    return timeout_limit


def _execution_result_from_payload(
        payload,
        *,
        reported_value: Any | None,
        timeout_limit_seconds: float | None,
) -> TSPSolverExecutionResult:
    status, value, child_runtime, error = payload
    return TSPSolverExecutionResult(
        status,
        value=value,
        reported_value=reported_value,
        runtime_seconds=child_runtime,
        error=error,
        timeout_limit_seconds=timeout_limit_seconds,
    )


def _worker(code: str, distance_matrix: np.ndarray, result_queue, best_tour_a, best_tour_b,
            active_tour_index) -> None:
    from dynagen.execution.sandbox import load_tsp_solver

    best_length = float("inf")

    def report_best_tour(tour: object) -> None:
        nonlocal best_length
        try:
            tour_arr = np.asarray(tour, dtype=float).reshape(-1)
            if tour_arr.size != distance_matrix.shape[0]:
                return None
            if not np.all(np.isfinite(tour_arr)):
                return None
            rounded = tour_arr.astype(np.int64)
            if not np.allclose(tour_arr, rounded):
                return None
            if rounded.size == 0:
                return None
            if int(rounded.min()) < 0 or int(rounded.max()) >= distance_matrix.shape[0]:
                return None
            if np.unique(rounded).size != distance_matrix.shape[0]:
                return None
            length = float(distance_matrix[rounded[-1], rounded[0]])
            length += float(distance_matrix[rounded[:-1], rounded[1:]].sum())
            if not np.isfinite(length) or length >= best_length:
                return None
            write_index = 1 if active_tour_index.value == 0 else 0
            target = best_tour_b if write_index == 1 else best_tour_a
            for index, node in enumerate(rounded):
                target[index] = float(node)
            active_tour_index.value = write_index
            best_length = length
        except Exception:
            return None

    start = time.perf_counter()
    try:
        tsp_solver = load_tsp_solver(code, best_tour_reporter=report_best_tour)
        tour = tsp_solver(distance_matrix.copy())
        runtime = time.perf_counter() - start
        result_queue.put(("ok", np.asarray(tour).tolist(), runtime, None))
    except Exception as exc:
        runtime = time.perf_counter() - start
        result_queue.put(("error", None, runtime, _short_error_message(exc)))


def _reported_tour(best_tour_a, best_tour_b, active_tour_index) -> Any | None:
    if active_tour_index.value == 0:
        return list(best_tour_a)
    if active_tour_index.value == 1:
        return list(best_tour_b)
    return None


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

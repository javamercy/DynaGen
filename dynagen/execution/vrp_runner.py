import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any, Literal

from dynagen.domain.vrp import evaluate_vrp_routes, VRPInstance

VRPSolverRunStatus = Literal["valid", "invalid", "timeout", "error"]
VRPSolverExecutionStatus = Literal["ok", "timeout", "error"]


@dataclass(frozen=True)
class VRPSolverExecutionResult:
    status: VRPSolverExecutionStatus
    value: Any = None
    reported_value: Any = None
    runtime_seconds: float = 0.0
    error: str | None = None
    timeout_limit_seconds: float | None = None


@dataclass(frozen=True)
class VRPSolverRunResult:
    status: VRPSolverRunStatus
    routes: list[list[int]]
    max_route_distance: float | None
    total_route_distance: float | None
    route_distances: list[float]
    visited_count: int
    runtime_seconds: float
    error: str | None = None
    partial: bool = False
    timeout_limit_seconds: float | None = None


def run_vrp_solver(
        code: str,
        instance: VRPInstance,
        *,
        timeout_seconds: float,
) -> VRPSolverRunResult:
    execution = execute_vrp_solver_code(
        code,
        instance,
        timeout_seconds=timeout_seconds,
    )
    if execution.status == "timeout":
        if execution.reported_value is not None:
            try:
                solution = evaluate_vrp_routes(instance, execution.reported_value)
                return _run_result("timeout", solution, execution, partial=True)
            except Exception as exc:
                error = f"{execution.error}; reported best routes invalid: {_short_error_message(exc)}"
                return _empty_run_result("timeout", execution, error=error)
        return _empty_run_result("timeout", execution)

    if execution.status != "ok":
        return _empty_run_result("error", execution)

    try:
        solution = evaluate_vrp_routes(instance, execution.value)
    except Exception as exc:
        return _empty_run_result("invalid", execution, error=_short_error_message(exc))
    return _run_result("valid", solution, execution, partial=False)


def execute_vrp_solver_code(
        code: str,
        instance: VRPInstance,
        *,
        timeout_seconds: float,
) -> VRPSolverExecutionResult:
    context = _multiprocessing_context()
    result_queue = context.Queue(maxsize=1)
    best_routes_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_worker,
        args=(code, instance, result_queue, best_routes_queue),
    )
    timeout_limit = float(timeout_seconds)
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
            reported_value=_reported_routes(best_routes_queue),
            timeout_limit_seconds=timeout_limit,
        )
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return VRPSolverExecutionResult(
            "timeout",
            reported_value=_reported_routes(best_routes_queue),
            runtime_seconds=runtime,
            error=f"VRP solver timed out after {runtime:.6g}s (timeout_seconds={timeout_limit:.6g})",
            timeout_limit_seconds=timeout_limit,
        )
    try:
        status, value, child_runtime, error = result_queue.get_nowait()
    except queue.Empty:
        if process.exitcode == 0:
            return VRPSolverExecutionResult(
                "error",
                reported_value=_reported_routes(best_routes_queue),
                runtime_seconds=runtime,
                error="VRP solver exited without returning a result",
                timeout_limit_seconds=timeout_limit,
            )
        return VRPSolverExecutionResult(
            "error",
            reported_value=_reported_routes(best_routes_queue),
            runtime_seconds=runtime,
            error=f"VRP solver process exited with code {process.exitcode}",
            timeout_limit_seconds=timeout_limit,
        )
    return VRPSolverExecutionResult(
        status,
        value=value,
        reported_value=_reported_routes(best_routes_queue),
        runtime_seconds=child_runtime,
        error=error,
        timeout_limit_seconds=timeout_limit,
    )


def _run_result(status: VRPSolverRunStatus, solution, execution: VRPSolverExecutionResult, *, partial: bool) -> VRPSolverRunResult:
    return VRPSolverRunResult(
        status,
        solution.routes,
        solution.max_route_distance,
        solution.total_route_distance,
        solution.route_distances,
        solution.visited_count,
        execution.runtime_seconds,
        execution.error,
        partial=partial,
        timeout_limit_seconds=execution.timeout_limit_seconds,
    )


def _empty_run_result(
        status: VRPSolverRunStatus,
        execution: VRPSolverExecutionResult,
        *,
        error: str | None = None,
) -> VRPSolverRunResult:
    return VRPSolverRunResult(
        status,
        [],
        None,
        None,
        [],
        0,
        execution.runtime_seconds,
        error if error is not None else execution.error,
        timeout_limit_seconds=execution.timeout_limit_seconds,
    )


def _get_worker_result_until_deadline(result_queue, process, *, start: float, timeout_seconds: float):
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


def _execution_result_from_payload(
        payload,
        *,
        reported_value: Any | None,
        timeout_limit_seconds: float,
) -> VRPSolverExecutionResult:
    status, value, child_runtime, error = payload
    return VRPSolverExecutionResult(
        status,
        value=value,
        reported_value=reported_value,
        runtime_seconds=child_runtime,
        error=error,
        timeout_limit_seconds=timeout_limit_seconds,
    )


def _worker(code: str, instance: VRPInstance, result_queue, best_routes_queue) -> None:
    from dynagen.execution.sandbox import load_vrp_solver

    def report_best_routes(routes: object) -> None:
        try:
            plain_routes = _plain_routes(routes)
            while True:
                try:
                    best_routes_queue.get_nowait()
                except queue.Empty:
                    break
            best_routes_queue.put_nowait(plain_routes)
        except Exception:
            return None

    start = time.perf_counter()
    try:
        solver = load_vrp_solver(code, best_routes_reporter=report_best_routes)
        routes = solver(instance.distance_matrix.copy(), int(instance.truck_count))
        runtime = time.perf_counter() - start
        result_queue.put(("ok", _plain_routes(routes), runtime, None))
    except Exception as exc:
        runtime = time.perf_counter() - start
        result_queue.put(("error", None, runtime, _short_error_message(exc)))


def _reported_routes(best_routes_queue) -> Any | None:
    latest = None
    while True:
        try:
            latest = best_routes_queue.get_nowait()
        except queue.Empty:
            return latest


def _plain_routes(routes: object) -> list[list[int]]:
    return [[int(node) for node in route] for route in routes]  # type: ignore[union-attr]


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

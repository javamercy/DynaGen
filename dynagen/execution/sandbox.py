import builtins
import collections
import heapq
import itertools
import math
import random
import time
from typing import Any, Callable

import numpy as np

try:
    import numba as _numba
except ImportError:
    _numba = None

from dynagen.candidates.validation import (
    ALLOWED_IMPORTS,
    validate_bbob_generated_code,
    validate_dvrp_generated_code,
    validate_vrp_generated_code,
    validate_generated_code,
)

ALLOWED_MODULES = {
    "numpy": np,
    "math": math,
    "random": random,
    "heapq": heapq,
    "itertools": itertools,
    "collections": collections,
    "time": time,
}
if _numba is not None:
    ALLOWED_MODULES["numba"] = _numba

SAFE_BUILTINS = {
    "abs": builtins.abs,
    "all": builtins.all,
    "any": builtins.any,
    "bool": builtins.bool,
    "dict": builtins.dict,
    "enumerate": builtins.enumerate,
    "Exception": builtins.Exception,
    "filter": builtins.filter,
    "float": builtins.float,
    "int": builtins.int,
    "iter": builtins.iter,
    "isinstance": builtins.isinstance,
    "len": builtins.len,
    "list": builtins.list,
    "map": builtins.map,
    "max": builtins.max,
    "min": builtins.min,
    "next": builtins.next,
    "object": builtins.object,
    "pow": builtins.pow,
    "range": builtins.range,
    "reversed": builtins.reversed,
    "round": builtins.round,
    "RuntimeError": builtins.RuntimeError,
    "set": builtins.set,
    "slice": builtins.slice,
    "sorted": builtins.sorted,
    "str": builtins.str,
    "sum": builtins.sum,
    "tuple": builtins.tuple,
    "TypeError": builtins.TypeError,
    "ValueError": builtins.ValueError,
    "zip": builtins.zip,
    "__build_class__": builtins.__build_class__,
    "__import__": None,
}


def load_tsp_solver(
        code: str,
        *,
        validate_static: bool = True,
        best_tour_reporter: Callable[[object], None] | None = None,
) -> Callable[..., object]:
    if validate_static:
        result = validate_generated_code(code)
        if not result.valid:
            raise ValueError(result.error)
    namespace = _sandbox_namespace(best_tour_reporter=best_tour_reporter)
    exec(compile(code, "<generated_candidate>", "exec"), namespace, namespace)
    solver = namespace.get("solve_tsp")
    if not callable(solver):
        raise ValueError("Generated code did not define callable solve_tsp")
    return solver


def load_bbob_optimizer(
        code: str,
        *,
        validate_static: bool = True,
        best_value_reporter: Callable[[object, object], None] | None = None,
) -> type:
    if validate_static:
        result = validate_bbob_generated_code(code)
        if not result.valid:
            raise ValueError(result.error)
    namespace = _sandbox_namespace(best_value_reporter=best_value_reporter)
    exec(compile(code, "<generated_candidate>", "exec"), namespace, namespace)
    optimizer = namespace.get("Optimizer")
    if not isinstance(optimizer, type):
        raise ValueError("Generated code did not define class Optimizer")
    return optimizer


def load_dvrp_policy(
        code: str,
        *,
        validate_static: bool = True,
) -> Callable[..., object]:
    if validate_static:
        result = validate_dvrp_generated_code(code)
        if not result.valid:
            raise ValueError(result.error)
    namespace = _sandbox_namespace()
    exec(compile(code, "<generated_candidate>", "exec"), namespace, namespace)
    policy = namespace.get("choose_next_customer")
    if not callable(policy):
        raise ValueError("Generated code did not define callable choose_next_customer")
    return policy


def load_vrp_solver(
        code: str,
        *,
        validate_static: bool = True,
        best_routes_reporter: Callable[[object], None] | None = None,
) -> Callable[..., object]:
    if validate_static:
        result = validate_vrp_generated_code(code)
        if not result.valid:
            raise ValueError(result.error)
    namespace = _sandbox_namespace(best_routes_reporter=best_routes_reporter)
    exec(compile(code, "<generated_candidate>", "exec"), namespace, namespace)
    solver = namespace.get("solve_vrp")
    if not callable(solver):
        raise ValueError("Generated code did not define callable solve_vrp")
    return solver


def _sandbox_namespace(
        *,
        best_tour_reporter: Callable[[object], None] | None = None,
        best_value_reporter: Callable[[object, object], None] | None = None,
        best_routes_reporter: Callable[[object], None] | None = None,
) -> dict[str, Any]:
    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "__name__": "generated_candidate",
        "np": np,
        "numpy": np,
        "math": math,
        "random": random,
        "report_best_tour": best_tour_reporter or _ignore_best_tour,
        "report_best": best_value_reporter or _ignore_best_value,
        "report_best_vrp": best_routes_reporter or _ignore_best_routes,
        "report_best_routes": best_routes_reporter or _ignore_best_routes,
        "heapq": heapq,
        "itertools": itertools,
        "collections": collections,
        "time": time,
    }
    if _numba is not None:
        namespace["numba"] = _numba
        namespace["njit"] = _numba.njit
        namespace["jit"] = _numba.jit
    return namespace


def _ignore_best_tour(tour: object) -> None:
    return None


def _ignore_best_value(value: object, x: object) -> None:
    return None


def _ignore_best_routes(routes: object) -> None:
    return None


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    if level != 0:
        raise ImportError("Relative imports are not allowed")
    root = name.split(".", 1)[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"Import not allowed: {name}")
    if root == "numba":
        if _numba is None:
            raise ImportError("Optional dependency numba is not installed")
        return __import__(name, globals, locals, fromlist, level)
    if name in ALLOWED_MODULES:
        return ALLOWED_MODULES[name]
    if root == "numpy" and name.startswith("numpy."):
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import not allowed: {name}")

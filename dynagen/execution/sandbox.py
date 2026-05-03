from __future__ import annotations

import builtins
import collections
import heapq
import itertools
import math
import random
import time
from types import FunctionType
from typing import Any, Callable

import numpy as np

from dynagen.candidates.validation import ALLOWED_IMPORTS, validate_generated_code

ALLOWED_MODULES = {
    "numpy": np,
    "math": math,
    "random": random,
    "heapq": heapq,
    "itertools": itertools,
    "collections": collections,
    "time": time,
}

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
    "__import__": None,
}


def load_tsp_solver(code: str, *, validate_static: bool = True) -> Callable[..., object]:
    if validate_static:
        result = validate_generated_code(code)
        if not result.valid:
            raise ValueError(result.error)
    namespace = _sandbox_namespace()
    exec(compile(code, "<generated_candidate>", "exec"), namespace, namespace)
    solver = namespace.get("solve_tsp")
    if not callable(solver):
        raise ValueError("Generated code did not define callable solve_tsp")
    return solver


def _sandbox_namespace() -> dict[str, Any]:
    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "__name__": "generated_candidate",
        "np": np,
        "numpy": np,
        "math": math,
        "random": random,
        "heapq": heapq,
        "itertools": itertools,
        "collections": collections,
        "time": time,
    }
    return namespace


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    if level != 0:
        raise ImportError("Relative imports are not allowed")
    root = name.split(".", 1)[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"Import not allowed: {name}")
    if name in ALLOWED_MODULES:
        return ALLOWED_MODULES[name]
    if root == "numpy" and name.startswith("numpy."):
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import not allowed: {name}")

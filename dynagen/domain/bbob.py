import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# 1, 6, 10, 15, 20
BBOB_GROUPS = {
    1: "separable",
    2: "separable",
    3: "separable",
    4: "separable",
    5: "separable",
    6: "low_moderate_conditioning",
    7: "low_moderate_conditioning",
    8: "low_moderate_conditioning",
    9: "low_moderate_conditioning",
    10: "high_conditioning_unimodal",
    11: "high_conditioning_unimodal",
    12: "high_conditioning_unimodal",
    13: "high_conditioning_unimodal",
    14: "high_conditioning_unimodal",
    15: "multimodal_strong_global_structure",
    16: "multimodal_strong_global_structure",
    17: "multimodal_strong_global_structure",
    18: "multimodal_strong_global_structure",
    19: "multimodal_strong_global_structure",
    20: "multimodal_weak_global_structure",
    21: "multimodal_weak_global_structure",
    22: "multimodal_weak_global_structure",
    23: "multimodal_weak_global_structure",
    24: "multimodal_weak_global_structure",
}

BBOB_NAMES = {
    1: "sphere",
    2: "ellipsoid_separable",
    3: "rastrigin_separable",
    4: "buche_rastrigin",
    5: "linear_slope",
    6: "attractive_sector",
    7: "step_ellipsoid",
    8: "rosenbrock_original",
    9: "rosenbrock_rotated",
    10: "ellipsoid_rotated",
    11: "discus",
    12: "bent_cigar",
    13: "sharp_ridge",
    14: "different_powers",
    15: "rastrigin_rotated",
    16: "weierstrass",
    17: "schaffer_f7",
    18: "schaffer_f7_ill_conditioned",
    19: "griewank_rosenbrock",
    20: "schwefel",
    21: "gallagher_101_peaks",
    22: "gallagher_21_peaks",
    23: "katsuura",
    24: "lunacek_bi_rastrigin",
}


@dataclass(frozen=True)
class Bounds:
    lb: np.ndarray
    ub: np.ndarray


@dataclass
class BBOBInstance:
    function_id: int
    instance_id: int
    dimension: int
    lower_bound: float = -5.0
    upper_bound: float = 5.0
    optimum_value: float = 0.0

    def __post_init__(self) -> None:
        self.function_id = int(self.function_id)
        self.instance_id = int(self.instance_id)
        self.dimension = int(self.dimension)
        if self.function_id not in BBOB_NAMES:
            raise ValueError(f"Unknown BBOB function id: {self.function_id}")
        if self.dimension < 1:
            raise ValueError("BBOB dimension must be positive")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("BBOB lower_bound must be below upper_bound")

    @property
    def name(self) -> str:
        return f"f{self.function_id:02d}_{BBOB_NAMES[self.function_id]}_i{self.instance_id}_d{self.dimension}"

    @property
    def group(self) -> str:
        return BBOB_GROUPS[self.function_id]

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            lb=np.full(self.dimension, self.lower_bound, dtype=float),
            ub=np.full(self.dimension, self.upper_bound, dtype=float),
        )

    def make_ioh_problem(self):
        return _ioh_problem(self.function_id, self.instance_id, self.dimension)

    def evaluate(self, x: object) -> float:
        return self.evaluate_ioh(x, self.make_ioh_problem())

    def evaluate_ioh(self, x: object, problem: object) -> float:
        point = np.asarray(x, dtype=float).reshape(-1)
        if point.size != self.dimension:
            raise ValueError(f"Expected point dimension {self.dimension}, got {point.size}")
        if not np.all(np.isfinite(point)):
            return float("inf")
        point = np.clip(point, self.lower_bound, self.upper_bound)
        value = float(problem(point.tolist()))
        if not math.isfinite(value):
            return float("inf")
        return value


class BudgetExceeded(RuntimeError):
    pass


class BudgetedBBOBObjective:
    def __init__(
            self,
            instance: BBOBInstance,
            *,
            budget: int,
            on_improvement: Callable[[float, np.ndarray], None] | None = None,
    ) -> None:
        self.instance = instance
        self.budget = int(budget)
        self._ioh_problem = instance.make_ioh_problem()
        self.bounds = _problem_bounds(self._ioh_problem, instance)
        self.evaluations = 0
        self.best_value = float("inf")
        self.best_x: np.ndarray | None = None
        self.history: list[float] = []
        self._on_improvement = on_improvement

    def __call__(self, x: object):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            return self._evaluate_one(arr)
        if arr.ndim == 2:
            return np.asarray([self._evaluate_one(row) for row in arr], dtype=float)
        raise ValueError("BBOB objective expects a 1D point or a 2D batch of points")

    def _evaluate_one(self, x: np.ndarray) -> float:
        if self.evaluations >= self.budget:
            raise BudgetExceeded("BBOB function evaluation budget exceeded")
        value = self.instance.evaluate_ioh(x, self._ioh_problem)
        self.evaluations += 1
        if value < self.best_value:
            self.best_value = float(value)
            self.best_x = np.asarray(x, dtype=float).reshape(-1).copy()
            if self._on_improvement is not None:
                self._on_improvement(self.best_value, self.best_x)
        self.history.append(self.best_value)
        return float(value)


def create_bbob_instances(
        *,
        function_ids: list[int] | tuple[int, ...],
        instance_ids: list[int] | tuple[int, ...],
        dimensions: list[int] | tuple[int, ...],
        bounds: list[float] | tuple[float, float] = (-5.0, 5.0),
) -> list[BBOBInstance]:
    _import_ioh()
    lower, upper = float(bounds[0]), float(bounds[1])
    instances: list[BBOBInstance] = []
    for dimension in dimensions:
        for function_id in function_ids:
            for instance_id in instance_ids:
                instances.append(_make_ioh_instance(function_id, instance_id, int(dimension), lower, upper))
    return instances


def _make_ioh_instance(function_id: int, instance_id: int, dimension: int, lower: float, upper: float) -> BBOBInstance:
    problem = _ioh_problem(function_id, instance_id, dimension)
    bounds = _problem_bounds(problem, None)
    optimum_value = _ioh_optimum_value(problem)
    return BBOBInstance(
        function_id=function_id,
        instance_id=instance_id,
        dimension=dimension,
        lower_bound=float(bounds.lb[0]) if bounds.lb.size else lower,
        upper_bound=float(bounds.ub[0]) if bounds.ub.size else upper,
        optimum_value=optimum_value,
    )


def _ioh_problem(function_id: int, instance_id: int, dimension: int):
    ioh = _import_ioh()
    try:
        problem_class = ioh.ProblemClass.BBOB
    except AttributeError as exc:
        raise RuntimeError("Installed ioh package does not expose ProblemClass.BBOB") from exc
    problem = ioh.get_problem(
        int(function_id),
        instance=int(instance_id),
        dimension=int(dimension),
        problem_class=problem_class,
    )
    reset = getattr(problem, "reset", None)
    if callable(reset):
        reset()
    return problem


def _import_ioh():
    try:
        import ioh
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "BBOB requires IOHexperimenter. Install it with: pip install ioh"
        ) from exc
    return ioh


def _problem_bounds(problem: object | None, instance: BBOBInstance | None) -> Bounds:
    dimension = instance.dimension if instance is not None else None
    if problem is not None:
        raw_bounds = getattr(problem, "bounds", None)
        lb = _bound_array(getattr(raw_bounds, "lb", None), dimension)
        ub = _bound_array(getattr(raw_bounds, "ub", None), dimension)
        if lb is not None and ub is not None:
            return Bounds(lb=lb, ub=ub)
    if instance is None:
        return Bounds(lb=np.zeros(0, dtype=float), ub=np.zeros(0, dtype=float))
    return instance.bounds


def _bound_array(value: object, dimension: int | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if dimension is not None and arr.size == 1:
        arr = np.full(dimension, float(arr[0]), dtype=float)
    if dimension is not None and arr.size != dimension:
        return None
    return arr


def _ioh_optimum_value(problem: object) -> float:
    optimum = getattr(problem, "optimum", None)
    for attr in ("y", "value"):
        value = getattr(optimum, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    meta_data = getattr(problem, "meta_data", None)
    for attr in ("yopt", "fopt", "optimum_y"):
        value = getattr(meta_data, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0

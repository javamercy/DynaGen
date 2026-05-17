import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VRP_TEST_SIZES = (10, 20, 50, 100, 200)


class VRPSolutionError(ValueError):
    """Raised when a generated VRP solver returns invalid routes."""


@dataclass
class VRPInstance:
    name: str
    coordinates: np.ndarray
    distance_matrix: np.ndarray
    truck_count: int
    reference_distance: float | None = None
    reference: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("VRP coordinates must have shape (n, 2)")
        if coordinates.shape[0] < 2:
            raise ValueError("VRP instances need one depot and at least one customer")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("VRP coordinates must be finite")

        distance_matrix = np.asarray(self.distance_matrix, dtype=float)
        n = coordinates.shape[0]
        if distance_matrix.shape != (n, n):
            raise ValueError("VRP distance_matrix must have shape (n, n)")
        if not np.all(np.isfinite(distance_matrix)) or np.any(distance_matrix < 0):
            raise ValueError("VRP distance_matrix must contain finite non-negative distances")

        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.truck_count = int(self.truck_count)
        if self.truck_count < 1:
            raise ValueError("VRP truck_count must be positive")
        if self.reference_distance is not None:
            self.reference_distance = float(self.reference_distance)
            if not np.isfinite(self.reference_distance) or self.reference_distance <= 0:
                raise ValueError("VRP reference_distance must be positive when provided")

    @property
    def dimension(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def customer_count(self) -> int:
        return self.dimension - 1


@dataclass(frozen=True)
class VRPSolutionResult:
    max_route_distance: float
    total_route_distance: float
    routes: list[list[int]]
    route_distances: list[float]
    visited_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_route_distance": float(self.max_route_distance),
            "total_route_distance": float(self.total_route_distance),
            "routes": [list(route) for route in self.routes],
            "route_distances": [float(distance) for distance in self.route_distances],
            "visited_count": int(self.visited_count),
        }


def load_vrp_instances(
        path: str | Path | None,
        *,
        pool_name: str,
        search_limit: int = 8,
        test_sizes: tuple[int, ...] | list[int] = VRP_TEST_SIZES,
        test_limit_per_size: int = 64,
) -> list[VRPInstance]:
    if not path:
        raise ValueError("VRP data.search_instances and data.test_instances must be specified")
    path = Path(path)
    if path.is_file():
        limit = search_limit if pool_name == "search_instances" else test_limit_per_size
        return _load_vrp_pickle(path, limit=limit)
    if not path.is_dir():
        raise ValueError(f"VRP data path does not exist: {path}")

    instances: list[VRPInstance] = []
    for size in test_sizes:
        file_path = path / f"instance_data_{int(size)}.pkl"
        if not file_path.exists():
            raise ValueError(f"Missing VRP test file: {file_path}")
        instances.extend(_load_vrp_pickle(file_path, limit=test_limit_per_size))
    return instances


def evaluate_vrp_routes(instance: VRPInstance, routes: object) -> VRPSolutionResult:
    normalized = _normalize_routes(routes, truck_count=instance.truck_count, dimension=instance.dimension)
    route_distances = [_route_distance(instance.distance_matrix, route) for route in normalized]
    total_route_distance = float(sum(route_distances))
    max_route_distance = float(max(route_distances))
    if not np.isfinite(max_route_distance) or max_route_distance <= 0:
        raise VRPSolutionError("VRP solver produced a non-finite route distance")
    return VRPSolutionResult(
        max_route_distance=max_route_distance,
        total_route_distance=total_route_distance,
        routes=normalized,
        route_distances=route_distances,
        visited_count=instance.customer_count,
    )


def _load_vrp_pickle(path: Path, *, limit: int | None = None) -> list[VRPInstance]:
    with path.open("rb") as handle:
        raw_items = pickle.load(handle)
    if not isinstance(raw_items, list):
        raise ValueError(f"VRP pickle must contain a list: {path}")
    selected = raw_items if limit is None else raw_items[:int(limit)]
    instances = [_instance_from_pickle_item(path, index, item) for index, item in enumerate(selected)]
    if limit is not None and len(instances) != int(limit):
        raise ValueError(f"VRP file {path} has {len(instances)} instances, expected {limit}")
    return instances


def _instance_from_pickle_item(path: Path, index: int, item: object) -> VRPInstance:
    if not isinstance(item, tuple) or len(item) != 3:
        raise ValueError(f"VRP pickle item {index} in {path} must be a 3-tuple")
    coordinates, distance_matrix, reference = item
    if not isinstance(reference, dict):
        raise ValueError(f"VRP reference item {index} in {path} must be a dict")
    routes = reference.get("routes") or []
    truck_count = len(routes)
    if truck_count < 1:
        raise ValueError(f"VRP reference item {index} in {path} has no routes")
    metadata = {
        "source": str(path),
        "source_file": path.name,
        "dataset_index": index,
    }
    return VRPInstance(
        name=f"{path.stem}_{index:03d}",
        coordinates=np.asarray(coordinates, dtype=float),
        distance_matrix=np.asarray(distance_matrix, dtype=float),
        truck_count=truck_count,
        reference_distance=_reference_distance(reference),
        reference=_plain_reference(reference),
        metadata=metadata,
    )


def _reference_distance(reference: dict[str, Any]) -> float | None:
    value = reference.get("max_distance")
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) and number > 0 else None


def _plain_reference(reference: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(value) for key, value in reference.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


def _normalize_routes(routes: object, *, truck_count: int, dimension: int) -> list[list[int]]:
    if not isinstance(routes, (list, tuple)):
        raise VRPSolutionError("solve_vrp must return a list of routes")
    if len(routes) != int(truck_count):
        raise VRPSolutionError(f"solve_vrp must return exactly {truck_count} routes")

    normalized: list[list[int]] = []
    visited: list[int] = []
    for route_value in routes:
        route = _normalize_route(route_value, dimension=dimension)
        normalized.append(route)
        visited.extend(node for node in route[1:-1] if node != 0)

    expected = list(range(1, int(dimension)))
    if sorted(visited) != expected:
        raise VRPSolutionError("VRP routes must visit every customer exactly once")
    return normalized


def _normalize_route(route_value: object, *, dimension: int) -> list[int]:
    try:
        values = np.asarray(route_value).reshape(-1)
    except Exception as exc:
        raise VRPSolutionError("Each VRP route must be a sequence of node ids") from exc
    if values.size < 2:
        raise VRPSolutionError("Each VRP route must include depot start and depot end")
    route: list[int] = []
    for value in values:
        numeric = float(value)
        if not np.isfinite(numeric) or int(numeric) != numeric:
            raise VRPSolutionError("VRP routes must contain integer node ids")
        node = int(numeric)
        if node < 0 or node >= int(dimension):
            raise VRPSolutionError(f"VRP route node {node} is outside the instance")
        route.append(node)
    if route[0] != 0 or route[-1] != 0:
        raise VRPSolutionError("Each VRP route must start and end at depot node 0")
    if any(node == 0 for node in route[1:-1]):
        raise VRPSolutionError("Depot node 0 may appear only at route boundaries")
    return route


def _route_distance(distance_matrix: np.ndarray, route: list[int]) -> float:
    total = 0.0
    for left, right in zip(route, route[1:]):
        total += float(distance_matrix[int(left), int(right)])
    return total

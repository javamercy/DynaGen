import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


PAPER_TEST_SIZES = (10, 20, 50, 100, 200)


class DVRPSimulationError(ValueError):
    """Raised when a generated policy cannot produce a valid DVRP rollout."""


@dataclass
class DVRPInstance:
    name: str
    coordinates: np.ndarray
    arrival_times: np.ndarray
    truck_count: int
    reference_makespan: float | None = None
    reference: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("DVRP coordinates must have shape (n, 2)")
        if coordinates.shape[0] < 2:
            raise ValueError("DVRP instances need one depot and at least one customer")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("DVRP coordinates must be finite")

        arrival_times = np.asarray(self.arrival_times, dtype=float).reshape(-1)
        if arrival_times.size != coordinates.shape[0]:
            raise ValueError("DVRP arrival_times length must match coordinates")
        if not np.all(np.isfinite(arrival_times)):
            raise ValueError("DVRP arrival_times must be finite")

        self.coordinates = coordinates
        self.arrival_times = arrival_times
        self.truck_count = int(self.truck_count)
        if self.truck_count < 1:
            raise ValueError("DVRP truck_count must be positive")
        if self.reference_makespan is not None:
            self.reference_makespan = float(self.reference_makespan)
            if not np.isfinite(self.reference_makespan) or self.reference_makespan <= 0:
                raise ValueError("DVRP reference_makespan must be positive when provided")

    @property
    def dimension(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def customer_count(self) -> int:
        return self.dimension - 1

    @property
    def depot(self) -> np.ndarray:
        return self.coordinates[0]


@dataclass(frozen=True)
class DVRPSimulationResult:
    makespan: float
    routes: list[list[int]]
    decisions: int
    waits: int
    completed_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "makespan": float(self.makespan),
            "routes": [list(route) for route in self.routes],
            "decisions": int(self.decisions),
            "waits": int(self.waits),
            "completed_count": int(self.completed_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DVRPSimulationResult":
        return cls(
            makespan=float(data["makespan"]),
            routes=[list(map(int, route)) for route in data["routes"]],
            decisions=int(data["decisions"]),
            waits=int(data["waits"]),
            completed_count=int(data["completed_count"]),
        )


DVRPPolicy = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int], object]


def load_dvrp_instances(
        path: str | Path | None,
        *,
        pool_name: str,
        search_limit: int = 8,
        test_sizes: tuple[int, ...] | list[int] = PAPER_TEST_SIZES,
        test_limit_per_size: int = 64,
) -> list[DVRPInstance]:
    if not path:
        raise ValueError("DVRP data.search_instances and data.test_instances must be specified")
    path = Path(path)
    if path.is_file():
        limit = search_limit if pool_name == "search_instances" else test_limit_per_size
        return _load_dvrp_pickle(path, limit=limit)
    if not path.is_dir():
        raise ValueError(f"DVRP data path does not exist: {path}")

    instances: list[DVRPInstance] = []
    for size in test_sizes:
        file_path = path / f"instance_data_{int(size)}.pkl"
        if not file_path.exists():
            raise ValueError(f"Missing DVRP test file: {file_path}")
        instances.extend(_load_dvrp_pickle(file_path, limit=test_limit_per_size))
    return instances


def simulate_dvrp_policy(
        instance: DVRPInstance,
        policy: DVRPPolicy,
        *,
        seed: int,
        budget: int,
) -> DVRPSimulationResult:
    requests = _RequestState(instance.coordinates, instance.arrival_times)
    trucks = [_TruckState(requests) for _ in range(instance.truck_count)]
    decisions = 0
    waits = 0

    while True:
        truck = trucks[int(np.argmin([item.time_left() for item in trucks]))]
        time_left = truck.time_left()
        if time_left > 0.0:
            for item in trucks:
                item.time_step(time_left)
            requests.time_step(time_left)

        if requests.is_done():
            break

        near_nodes = requests.near_nodes(truck.cur_node())
        mask = np.isin(near_nodes, requests.completed)
        for other in trucks:
            if other is not truck:
                mask |= np.isin(near_nodes, other.route[-1])
        available_nodes = near_nodes[np.invert(mask)]

        if available_nodes.size == 0:
            truck.wait()
            waits += 1
            continue

        customer_positions = np.asarray([instance.coordinates[node].copy() for node in available_nodes], dtype=float)
        truck_positions = np.asarray([item.position.copy() for item in trucks], dtype=float)
        choice = policy(
            truck.position.copy(),
            instance.depot.copy(),
            truck_positions,
            customer_positions,
            float(requests.current_time),
            int(seed) + decisions,
            int(budget),
        )
        decisions += 1

        if choice is None:
            truck.wait()
            waits += 1
            continue

        node_index = _as_customer_index(choice, available_nodes.size)
        truck.set_dest(int(available_nodes[node_index]))

    for truck in trucks:
        truck.go_home()

    routes = [list(map(int, truck.route)) for truck in trucks]
    _validate_routes(routes, instance.dimension)
    makespans = [truck.tour_cost() for truck in trucks]
    makespan = float(max(makespans))
    if not np.isfinite(makespan) or makespan <= 0:
        raise DVRPSimulationError("DVRP policy produced a non-finite makespan")
    return DVRPSimulationResult(
        makespan=makespan,
        routes=routes,
        decisions=decisions,
        waits=waits,
        completed_count=len(requests.completed) - 1,
    )


def _load_dvrp_pickle(path: Path, *, limit: int | None = None) -> list[DVRPInstance]:
    with path.open("rb") as handle:
        raw_items = pickle.load(handle)
    if not isinstance(raw_items, list):
        raise ValueError(f"DVRP pickle must contain a list: {path}")
    selected = raw_items if limit is None else raw_items[:int(limit)]
    instances = []
    for index, item in enumerate(selected):
        instances.append(_instance_from_pickle_item(path, index, item))
    if limit is not None and len(instances) != int(limit):
        raise ValueError(f"DVRP file {path} has {len(instances)} instances, expected {limit}")
    return instances


def _instance_from_pickle_item(path: Path, index: int, item: object) -> DVRPInstance:
    if not isinstance(item, tuple) or len(item) != 3:
        raise ValueError(f"DVRP pickle item {index} in {path} must be a 3-tuple")
    coordinates, arrival_times, reference = item
    if not isinstance(reference, dict):
        raise ValueError(f"DVRP reference item {index} in {path} must be a dict")
    routes = reference.get("routes") or []
    truck_count = len(routes)
    reference_makespan = reference.get("max_distance")
    metadata = {
        "source": str(path),
        "source_file": path.name,
        "dataset_index": index,
    }
    return DVRPInstance(
        name=f"{path.stem}_{index:03d}",
        coordinates=np.asarray(coordinates, dtype=float),
        arrival_times=np.asarray(arrival_times, dtype=float),
        truck_count=truck_count,
        reference_makespan=None if reference_makespan is None else float(reference_makespan),
        reference=_plain_reference(reference),
        metadata=metadata,
    )


def _plain_reference(reference: dict[str, Any]) -> dict[str, Any]:
    plain: dict[str, Any] = {}
    for key, value in reference.items():
        if isinstance(value, np.ndarray):
            plain[str(key)] = value.tolist()
        elif isinstance(value, list):
            plain[str(key)] = [_plain_value(item) for item in value]
        elif isinstance(value, tuple):
            plain[str(key)] = [_plain_value(item) for item in value]
        else:
            plain[str(key)] = _plain_value(value)
    return plain


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


def _as_customer_index(value: object, size: int) -> int:
    try:
        arr = np.asarray(value)
        if arr.shape not in {(), (1,)}:
            raise ValueError
        numeric = float(arr.reshape(-1)[0])
    except Exception as exc:
        raise DVRPSimulationError("Policy must return an integer customer index or None") from exc
    if not np.isfinite(numeric) or int(numeric) != numeric:
        raise DVRPSimulationError("Policy returned a non-integer customer index")
    index = int(numeric)
    if index < 0 or index >= int(size):
        raise DVRPSimulationError(f"Policy returned customer index {index}, but {size} are available")
    return index


def _validate_routes(routes: list[list[int]], dimension: int) -> None:
    visited = [node for route in routes for node in route if node != 0]
    expected = list(range(1, int(dimension)))
    if sorted(visited) != expected:
        raise DVRPSimulationError("DVRP rollout did not visit every customer exactly once")
    for route in routes:
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            raise DVRPSimulationError("Each DVRP truck route must start and end at the depot")


class _RequestState:
    def __init__(self, locations: np.ndarray, arrival_times: np.ndarray) -> None:
        self.locations = np.asarray(locations, dtype=float)
        self.arrival_times = np.asarray(arrival_times, dtype=float)
        self.current_time = 0.0
        self.incomplete = [index for index in range(1, self.locations.shape[0])]
        self.completed = [0]
        self._available = []
        self._unavailable = []
        for index, arrival in enumerate(self.arrival_times):
            if arrival <= 0.0:
                self._available.append(index)
            else:
                self._unavailable.append(index)
        self.neighborhood_matrix = _neighborhood_matrix(self.locations)

    def time_step(self, step: float) -> None:
        self.current_time += float(step)
        # Match the paper implementation exactly, including list mutation during iteration.
        for index in self._unavailable:
            if self.arrival_times[index] < self.current_time:
                self._available.append(index)
                self._unavailable.remove(index)

    def near_nodes(self, node: int) -> np.ndarray:
        near_nodes = self.neighborhood_matrix[int(node)]
        mask = np.isin(near_nodes, self._available)
        return near_nodes[mask]

    def currently_at(self, node: int, from_time: float, to_time: float) -> None:
        node = int(node)
        if node not in self.incomplete:
            return
        if to_time < self.arrival_times[node]:
            return
        self.incomplete.remove(node)
        self.completed.append(node)

    def is_done(self) -> bool:
        if self.current_time > 1000:
            return True
        return len(self.incomplete) == 0


class _TruckState:
    def __init__(self, requests: _RequestState) -> None:
        self.route = [0]
        self._time_moving = 0.0
        self._time_waited = 0.0
        self.requests = requests
        self.locations = requests.locations
        self.depot = self.locations[0]
        self.current_time = 0.0
        self.position = np.array(self.depot, dtype=float)
        self._dest: np.ndarray | None = None
        self._wait_left = 0.0

    def set_dest(self, node: int) -> None:
        if self._dest is not None or self._wait_left != 0.0:
            raise DVRPSimulationError("Truck received a new destination while busy")
        self.requests.currently_at(self.cur_node(), self.current_time, self.current_time)
        if self.cur_node() == node:
            self.wait()
            return
        self.route.append(int(node))
        self._dest = np.array(self.locations[int(node)], dtype=float)

    def time_left(self) -> float:
        if self._dest is None:
            return float(self._wait_left)
        return float(np.linalg.norm(self.position - self._dest))

    def cur_node(self) -> int:
        if self._dest is not None:
            raise DVRPSimulationError("Cannot ask for current node while truck is moving")
        return int(self.route[-1])

    def wait(self) -> None:
        if self._dest is not None:
            raise DVRPSimulationError("Moving truck cannot wait")
        self._wait_left = 0.1

    def time_step(self, step: float) -> None:
        step = float(step)
        self.current_time += step
        if self._dest is None:
            if step > self._wait_left + 1e-8:
                raise DVRPSimulationError("Wait step exceeds remaining wait time")
            self.requests.currently_at(self.cur_node(), self.current_time - step, self.current_time)
            self._wait_left -= step
            self._time_waited += step
            if self._wait_left <= 1e-8:
                self._wait_left = 0.0
            return

        distance = float(np.linalg.norm(self.position - self._dest))
        if step > distance + 1e-8:
            raise DVRPSimulationError("Travel step exceeds remaining distance")
        self.position += (self._dest - self.position) * (step / distance)
        self._time_moving += step
        if abs(step - distance) <= 1e-8:
            self.position = np.array(self._dest, dtype=float)
            self._dest = None
            self.requests.currently_at(self.cur_node(), self.current_time, self.current_time)

    def go_home(self) -> None:
        if self._dest is None:
            self.current_time += self._wait_left
            self.requests.currently_at(self.cur_node(), self.current_time - self._wait_left, self.current_time)
            self._time_waited += self._wait_left
            self._wait_left = 0.0
        else:
            distance = float(np.linalg.norm(self.position - self._dest))
            self.current_time += distance
            self.position = np.array(self._dest, dtype=float)
            self._time_moving += distance
            self._dest = None
            self.requests.currently_at(self.cur_node(), self.current_time, self.current_time)
        self.route.append(0)
        self._time_moving += float(np.linalg.norm(self.position - self.depot))
        self.position = np.array(self.depot, dtype=float)

    def tour_cost(self) -> float:
        return float(self._time_moving + self._time_waited)


def _neighborhood_matrix(locations: np.ndarray) -> np.ndarray:
    n = int(locations.shape[0])
    matrix = np.zeros((n, n), dtype=int)
    for index in range(n):
        distances = np.linalg.norm(locations - locations[index], axis=1)
        matrix[index] = np.argsort(distances)
    return matrix

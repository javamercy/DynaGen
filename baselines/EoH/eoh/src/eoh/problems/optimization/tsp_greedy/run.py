import math
import signal
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np

from .get_instance import GetData
from .prompts import GetPrompts


DEFAULT_TIMEOUT_GAP = 1000000.0


class EvaluationTimeoutError(TimeoutError):
    pass


class TSPCONST():
    def __init__(self, paras=None) -> None:
        self.ndelay = 1
        self.problem_size = 50
        self.neighbor_size = np.minimum(50, self.problem_size)
        self.n_instance = 8
        self.running_time = 10
        self.test_instance_data = []
        self.timeout_gap = float(getattr(paras, "tsp_timeout_gap", DEFAULT_TIMEOUT_GAP))
        self.prompts = GetPrompts()

        search_instances = getattr(paras, "tsp_search_instances", None)
        test_instances = getattr(paras, "tsp_test_instances", None)
        if search_instances:
            self.instance_data = load_tsplib_pool(search_instances)
            self.n_instance = len(self.instance_data)
            if test_instances:
                self.test_instance_data = load_tsplib_pool(test_instances)
        else:
            getData = GetData(self.n_instance, self.problem_size)
            self.instance_data = getData.generate_instances()

    def tour_cost(self, distance_matrix, solution):
        nodes = np.asarray(solution, dtype=int)
        next_nodes = np.roll(nodes, -1)
        return float(np.asarray(distance_matrix, dtype=float)[nodes, next_nodes].sum())

    def generate_neighborhood_matrix(self, instance, distance_matrix=None):
        if distance_matrix is not None:
            return np.argsort(np.asarray(distance_matrix, dtype=float), axis=1).astype(int)

        instance = np.array(instance)
        n = len(instance)
        neighborhood_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)
            neighborhood_matrix[i] = np.argsort(distances)
        return neighborhood_matrix

    def greedy(self, eva, instance_data=None, collect_records=False, pool_name="search_instances"):
        dataset = self.instance_data if instance_data is None else instance_data
        scores = []
        tour_lengths = []
        records = []

        for raw_instance in dataset:
            instance_record = normalise_instance_record(raw_instance)
            instance = instance_record["coordinates"]
            distance_matrix = instance_record["distance_matrix"]
            optimal_length = instance_record.get("optimal_length")
            problem_size = int(distance_matrix.shape[0])
            neighbor_size = problem_size

            neighbor_matrix = self.generate_neighborhood_matrix(instance, distance_matrix)
            destination_node = 0
            current_node = 0
            route = np.zeros(problem_size, dtype=int)

            for i in range(1, problem_size - 1):
                near_nodes = neighbor_matrix[current_node][1:]
                mask = ~np.isin(near_nodes, route[:i])
                unvisited_near_nodes = near_nodes[mask]
                unvisited_near_size = np.minimum(neighbor_size, unvisited_near_nodes.size)
                unvisited_near_nodes = unvisited_near_nodes[:unvisited_near_size]

                next_node = eva.select_next_node(current_node, destination_node, unvisited_near_nodes, distance_matrix)
                try:
                    next_node = int(np.asarray(next_node).reshape(-1)[0])
                except Exception:
                    return self._invalid_result(records, collect_records, pool_name, instance_record,
                                                "selected node is not an integer")

                if next_node in route[:i] or next_node not in unvisited_near_nodes:
                    return self._invalid_result(records, collect_records, pool_name, instance_record,
                                                "selected node is not an unvisited node")

                current_node = next_node
                route[i] = current_node

            mask = ~np.isin(np.arange(problem_size), route[:problem_size - 1])
            last_node = np.arange(problem_size)[mask]
            current_node = int(last_node[0])
            route[problem_size - 1] = current_node

            tour_length = self.tour_cost(distance_matrix, route)
            gap = compute_gap(tour_length, optimal_length)
            score = gap if gap is not None else tour_length
            scores.append(score)
            tour_lengths.append(tour_length)
            records.append({
                "instance": instance_record["name"],
                "pool": pool_name,
                "dimension": problem_size,
                "source": instance_record.get("source"),
                "status": "valid",
                "tour_length": tour_length,
                "reference_length": optimal_length,
                "reference_kind": "optimal" if optimal_length is not None else None,
                "gap": gap,
                "route": route.astype(int).tolist(),
            })

        fitness = float(np.average(scores)) if scores else None
        if collect_records:
            gaps = [record["gap"] for record in records if record["gap"] is not None and math.isfinite(record["gap"])]
            return {
                "status": "valid" if records else "invalid",
                "fitness": fitness,
                "metrics": {
                    "pool": pool_name,
                    "runs": len(records),
                    "valid_count": len(records),
                    "mean_gap": float(np.average(gaps)) if gaps else None,
                    "mean_tour_length": float(np.average(tour_lengths)) if tour_lengths else None,
                    "records": records,
                },
            }
        return fitness

    def evaluate(self, code_string, timeout_seconds=None):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heuristic_module = self._module_from_code(code_string)
                result = self.evaluate_pool(
                    heuristic_module,
                    self.instance_data,
                    pool_name="search_instances",
                    timeout_seconds=timeout_seconds,
                )
                return result["fitness"]
        except Exception:
            return self.timeout_gap

    def evaluate_pool(self, heuristic_module, instance_data, pool_name, timeout_seconds=None):
        records = []
        deadline = None
        if timeout_seconds is not None and timeout_seconds > 0:
            deadline = time.monotonic() + float(timeout_seconds)

        for raw_instance in instance_data:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                records.append(build_nonvalid_record(raw_instance, pool_name, "timeout", "evaluation timed out"))
                continue

            try:
                with evaluation_timeout(remaining):
                    result = self.greedy(
                        heuristic_module,
                        instance_data=[raw_instance],
                        collect_records=True,
                        pool_name=pool_name,
                    )
                records.extend(result["metrics"].get("records", []))
            except EvaluationTimeoutError as exc:
                records.append(build_nonvalid_record(raw_instance, pool_name, "timeout", str(exc)))
            except Exception as exc:
                records.append(build_nonvalid_record(raw_instance, pool_name, "error", str(exc)))

        metrics = summarise_records(records, pool_name, self.timeout_gap)
        status = candidate_status_from_records(records)
        return {
            "status": status,
            "fitness": metrics["penalized_mean_gap"],
            "error_details": first_error(records),
            "metrics": metrics,
        }

    def evaluate_test(self, code_string, timeout_seconds=None):
        if not self.test_instance_data:
            return {
                "status": "skipped",
                "fitness": None,
                "error_details": "No TSP test instances configured",
                "metrics": {"pool": "test_instances", "runs": 0, "valid_count": 0, "records": []},
            }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heuristic_module = self._module_from_code(code_string)
                return self.evaluate_pool(
                    heuristic_module,
                    self.test_instance_data,
                    pool_name="test_instances",
                    timeout_seconds=timeout_seconds,
                )
        except Exception as exc:
            return {
                "status": "error",
                "fitness": self.timeout_gap,
                "error_details": str(exc),
                "metrics": {"pool": "test_instances", "runs": 0, "valid_count": 0, "records": []},
            }

    def _module_from_code(self, code_string):
        heuristic_module = types.ModuleType("heuristic_module")
        exec(code_string, heuristic_module.__dict__)
        sys.modules[heuristic_module.__name__] = heuristic_module
        return heuristic_module

    def _invalid_result(self, records, collect_records, pool_name, instance_record, error):
        if not collect_records:
            return None
        records.append({
            "instance": instance_record["name"],
            "pool": pool_name,
            "dimension": int(instance_record["distance_matrix"].shape[0]),
            "source": instance_record.get("source"),
            "status": "invalid",
            "tour_length": None,
            "reference_length": instance_record.get("optimal_length"),
            "reference_kind": "optimal" if instance_record.get("optimal_length") is not None else None,
            "gap": None,
            "error": error,
        })
        return {
            "status": "invalid",
            "fitness": None,
            "error_details": error,
            "metrics": {
                "pool": pool_name,
                "runs": len(records),
                "valid_count": sum(1 for record in records if record["status"] == "valid"),
                "mean_gap": None,
                "mean_tour_length": None,
                "records": records,
            },
        }


def normalise_instance_record(raw_instance):
    if isinstance(raw_instance, dict):
        record = dict(raw_instance)
        record["distance_matrix"] = np.asarray(record["distance_matrix"], dtype=float)
        if record.get("coordinates") is not None:
            record["coordinates"] = np.asarray(record["coordinates"], dtype=float)
        return record

    coordinates, distance_matrix = raw_instance
    return {
        "name": "generated",
        "coordinates": np.asarray(coordinates, dtype=float),
        "distance_matrix": np.asarray(distance_matrix, dtype=float),
        "optimal_length": None,
        "source": None,
    }


def compute_gap(tour_length, optimal_length):
    if optimal_length is None:
        return None
    optimal = float(optimal_length)
    if not math.isfinite(optimal) or optimal <= 0:
        return None
    return 100.0 * (float(tour_length) - optimal) / optimal


class evaluation_timeout:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.previous_handler = None

    def __enter__(self):
        if self.timeout_seconds is None or self.timeout_seconds <= 0:
            return self
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(self.timeout_seconds))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.timeout_seconds is not None and self.timeout_seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def timeout_handler(signum, frame):
    raise EvaluationTimeoutError("evaluation timed out")


def build_nonvalid_record(raw_instance, pool_name, status, error):
    instance_record = normalise_instance_record(raw_instance)
    return {
        "instance": instance_record["name"],
        "pool": pool_name,
        "dimension": int(instance_record["distance_matrix"].shape[0]),
        "source": instance_record.get("source"),
        "status": status,
        "tour_length": None,
        "reference_length": instance_record.get("optimal_length"),
        "reference_kind": "optimal" if instance_record.get("optimal_length") is not None else None,
        "gap": None,
        "error": error,
    }


def summarise_records(records, pool_name, timeout_gap=DEFAULT_TIMEOUT_GAP):
    gaps = [float(record["gap"]) for record in records if record.get("gap") is not None and math.isfinite(record["gap"])]
    tour_lengths = [float(record["tour_length"]) for record in records if record.get("tour_length") is not None]
    scores = [record_score(record, timeout_gap) for record in records]
    return {
        "pool": pool_name,
        "runs": len(records),
        "valid_count": sum(1 for record in records if record["status"] == "valid"),
        "timeout_count": sum(1 for record in records if record["status"] == "timeout"),
        "runtime_error_count": sum(1 for record in records if record["status"] == "error"),
        "invalid_count": sum(1 for record in records if record["status"] == "invalid"),
        "mean_gap": float(np.average(gaps)) if gaps else None,
        "penalized_mean_gap": float(np.average(scores)) if scores else float(timeout_gap),
        "timeout_gap": float(timeout_gap),
        "mean_tour_length": float(np.average(tour_lengths)) if tour_lengths else None,
        "records": records,
    }


def record_score(record, timeout_gap):
    gap = record.get("gap")
    if gap is not None and math.isfinite(gap):
        return float(gap)
    tour_length = record.get("tour_length")
    if record.get("status") == "valid" and tour_length is not None and math.isfinite(tour_length):
        return float(tour_length)
    return float(timeout_gap)


def candidate_status_from_records(records):
    if not records:
        return "invalid"
    if any(record["status"] == "timeout" for record in records):
        return "timeout"
    if any(record["status"] == "error" for record in records):
        return "error"
    if any(record["status"] == "invalid" for record in records):
        return "invalid"
    return "valid"


def first_error(records):
    for record in records:
        if record["status"] != "valid":
            return f"{record['status']} on {record['instance']}: {record.get('error')}"
    return None


def load_tsplib_pool(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.suffix.lower() == ".tsp")
    else:
        files = [path]
    if not files:
        raise ValueError(f"No .tsp files found in {path}")
    return [load_tsplib_file(file) for file in files]


def load_tsplib_file(path):
    path = Path(path)
    headers = {}
    coordinates = []
    weights = []
    section = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper in {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION"}:
            section = upper
            continue
        if section == "NODE_COORD_SECTION":
            node_id, x, y = line.split()[:3]
            coordinates.append((int(node_id), float(x), float(y)))
            continue
        if section == "EDGE_WEIGHT_SECTION":
            weights.extend(float(value) for value in line.split())
            continue
        key, value = parse_tsplib_header(line)
        headers[key.upper()] = value

    name = headers.get("NAME", path.stem)
    dimension = int(headers["DIMENSION"])
    edge_weight_type = headers.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    optimal_length = float(headers["OPTIMAL"]) if "OPTIMAL" in headers else None

    if edge_weight_type in {"EUC_2D", "CEIL_2D"}:
        coordinates_arr = coordinates_to_array(coordinates, dimension)
        distance_matrix = euclidean_distance_matrix(coordinates_arr, edge_weight_type)
    elif edge_weight_type == "EXPLICIT":
        coordinates_arr = None
        distance_matrix = np.asarray(weights, dtype=float).reshape((dimension, dimension))
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE in {path}: {edge_weight_type}")

    return {
        "name": name,
        "coordinates": coordinates_arr,
        "distance_matrix": distance_matrix,
        "optimal_length": optimal_length,
        "source": str(path),
    }


def parse_tsplib_header(line):
    if ":" in line:
        key, value = line.split(":", 1)
        return key.strip(), value.strip()
    key, value = line.split(maxsplit=1)
    return key.strip(), value.strip()


def coordinates_to_array(coordinates, dimension):
    if len(coordinates) != dimension:
        raise ValueError("NODE_COORD_SECTION count does not match DIMENSION")
    coordinates = sorted(coordinates, key=lambda item: item[0])
    return np.array([[x, y] for _, x, y in coordinates], dtype=float)


def euclidean_distance_matrix(coordinates, edge_weight_type):
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    if edge_weight_type == "EUC_2D":
        distances = np.floor(distances + 0.5)
    elif edge_weight_type == "CEIL_2D":
        distances = np.ceil(distances)
    np.fill_diagonal(distances, 0.0)
    return distances.astype(float)

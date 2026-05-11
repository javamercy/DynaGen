"""Multi-objective LLaMEA example on a synthetic TSP variant.

Search uses the same synthetic instance as the original LLaMEA example:
``generate_tsp_test(seed=69, size=32)`` by default. The final selected
solution is then evaluated on the shared TSPLIB test pool under
``data/tsp/test_instances``.
"""

import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynagen.domain import load_tsplib_file
from llamea import LLaMEA, Solution
from llamea.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM
from llamea.loggers import ExperimentLogger
from llamea.multi_objective_fitness import Fitness
from llamea.pareto_archive import ParetoArchive
from llamea.utils import prepare_namespace


@dataclass
class Location:
    id: int
    x: float
    y: float
    weight: float

    def vectorise(self):
        return [self.id, self.x, self.y, self.weight]

    def __repr__(self):
        return (
            f"Location(id: {self.id}, coordinates: ({self.x}, {self.y}), "
            f"weight: {self.weight})"
        )


@dataclass
class TSPCase:
    name: str
    depot: Location
    customers: list[Location]
    distance_matrix: np.ndarray | None = None
    optimal_length: float | None = None
    source: str | None = None


SEARCH_CASE: TSPCase | None = None
DEFAULT_TEST_TIMEOUT_SECONDS = 60.0


class TestEvaluationTimeoutError(TimeoutError):
    pass


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _prepare_output_dir() -> Path:
    output_dir = Path(os.getenv("LLAMEA_OUTPUT_DIR", str(PROJECT_ROOT / "runs" / "tsp")))
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    print(f"LLaMEA TSP output directory: {output_dir}")
    return output_dir


def _build_llm(model: str):
    provider = "openai"
    if provider == "ollama":
        return Ollama_LLM(model)
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAI_LLM(api_key, model)
    api_key = os.getenv("GOOGLE_API_KEY")
    return Gemini_LLM(api_key, model)


def generate_tsp_test(seed: Optional[int] = None, size: int = 10):
    """Generate a depot and customer set for the synthetic TSP task."""
    rng = random.Random(seed) if seed is not None else random
    depot = Location(0, 50, 50, 0)
    customers: list[Location] = []
    for id in range(size):
        x = rng.randint(0, 100)
        y = rng.randint(0, 100)
        weight = rng.randint(10, 35)
        customers.append(Location(id + 1, x, y, weight))
    return depot, customers


def build_generated_case(seed: int, size: int) -> TSPCase:
    depot, customers = generate_tsp_test(seed=seed, size=size)
    return TSPCase(
        name=f"llamea_seed{seed}_size{size}",
        depot=depot,
        customers=customers,
        source=f"synthetic:llamea:{seed}:{size}",
    )


def load_test_cases(path: str | Path) -> list[TSPCase]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    files = sorted(path.iterdir()) if path.is_dir() else [path]
    tsp_files = [item for item in files if item.suffix.lower() == ".tsp"]
    if not tsp_files:
        raise ValueError(f"No .tsp files found in {path}")

    cases = []
    for file in tsp_files:
        instance = load_tsplib_file(file)
        if instance.coordinates is None:
            raise ValueError(f"TSP instance {file} does not include coordinates")
        depot = Location(0, instance.coordinates[0, 0], instance.coordinates[0, 1], 0)
        customers = [
            Location(index, instance.coordinates[index, 0], instance.coordinates[index, 1], 1)
            for index in range(1, instance.dimension)
        ]
        cases.append(
            TSPCase(
                name=instance.name,
                depot=depot,
                customers=customers,
                distance_matrix=instance.distance_matrix,
                optimal_length=instance.optimal_length,
                source=instance.metadata.get("source"),
            )
        )
    return cases


def evaluate(solution: Solution, explogger: Optional[ExperimentLogger] = None):
    """Evaluate generated solver code on the synthetic search TSP case."""
    if SEARCH_CASE is None:
        raise RuntimeError("SEARCH_CASE was not initialized")

    global_ns, issues = prepare_namespace(
        solution.code,
        ["numpy", "pymoo", "typing", "scipy"],
        explogger,
    )
    global_ns["Location"] = Location
    local_ns = {}
    feedback = ""
    if issues:
        feedback += f"Import issues: {issues}. "
        print(f"Potential Issues {issues}.")

    try:
        compiled = compile(solution.code, "<llm_code>", "exec")
        exec(compiled, global_ns, local_ns)
        path_index = _run_solution(local_ns[solution.name], SEARCH_CASE)
        record = score_route(SEARCH_CASE, path_index, pool_name="search_instances")
    except Exception as exc:
        solution.set_scores(
            Fitness({"Distance": float("inf"), "Fuel": float("inf")}),
            feedback=f"Runtime error: {exc}",
            error=exc,
        )
        return solution

    if record["status"] != "valid":
        solution.set_scores(
            Fitness({"Distance": float("inf"), "Fuel": float("inf")}),
            feedback=record.get("error", "Invalid route"),
        )
        return solution

    fitness = Fitness({"Distance": record["tour_length"], "Fuel": record["fuel"]})
    solution.add_metadata("search_record", record)
    solution.set_scores(fitness, feedback=f"Fitness {fitness} for path {record['route']}")
    return solution


def evaluate_test_solution(
        solution: Solution,
        test_cases: list[TSPCase],
        timeout_seconds: float | None = None,
) -> dict:
    global_ns, issues = prepare_namespace(
        solution.code,
        ["numpy", "pymoo", "typing", "scipy"],
        None,
    )
    global_ns["Location"] = Location
    local_ns = {}

    try:
        with evaluation_timeout(timeout_seconds):
            compiled = compile(solution.code, "<llm_code>", "exec")
            exec(compiled, global_ns, local_ns)
            cls = local_ns[solution.name]
    except TestEvaluationTimeoutError as exc:
        records = [error_record(case, "test_instances", "timeout", str(exc)) for case in test_cases]
        return summarize_records(records, import_issues=issues, timeout_seconds=timeout_seconds)
    except Exception as exc:
        records = [error_record(case, "test_instances", "error", str(exc)) for case in test_cases]
        return summarize_records(records, import_issues=issues, timeout_seconds=timeout_seconds)

    records = []
    for case in test_cases:
        start = time.monotonic()
        try:
            with evaluation_timeout(timeout_seconds):
                path_index = _run_solution(cls, case)
            record = score_route(case, path_index, pool_name="test_instances")
            record["runtime_seconds"] = time.monotonic() - start
            records.append(record)
        except TestEvaluationTimeoutError as exc:
            records.append(
                error_record(
                    case,
                    "test_instances",
                    "timeout",
                    str(exc),
                    runtime_seconds=time.monotonic() - start,
                )
            )
        except Exception as exc:
            records.append(
                error_record(
                    case,
                    "test_instances",
                    "error",
                    str(exc),
                    runtime_seconds=time.monotonic() - start,
                )
            )
    return summarize_records(records, import_issues=issues, timeout_seconds=timeout_seconds)


def _run_solution(cls, case: TSPCase):
    return cls(
        case.depot.vectorise(),
        [customer.vectorise() for customer in case.customers],
    )()


def score_route(case: TSPCase, path_index, *, pool_name: str) -> dict:
    route = normalize_route(path_index)
    valid_ids = {customer.id for customer in case.customers}
    if len(route) != len(case.customers):
        return error_record(case, pool_name, "invalid", "Path length does not match number of customers")
    if len(set(route)) != len(route):
        return error_record(case, pool_name, "invalid", "Path contains duplicate customer indices")
    if not all(idx in valid_ids for idx in route):
        return error_record(case, pool_name, "invalid", "Path contains invalid customer indices")

    tour_length = route_distance(case, route)
    gap = compute_gap(tour_length, case.optimal_length)
    record = {
        "instance": case.name,
        "pool": pool_name,
        "dimension": len(case.customers) + 1,
        "source": case.source,
        "status": "valid",
        "tour_length": tour_length,
        "reference_length": case.optimal_length,
        "reference_kind": "optimal" if case.optimal_length is not None else None,
        "gap": gap,
        "route": route,
    }
    if pool_name == "search_instances":
        record["fuel"] = route_fuel(case, route)
    return record


def normalize_route(path_index) -> list[int]:
    if isinstance(path_index, np.ndarray):
        path_index = path_index.tolist()
    if not isinstance(path_index, (list, tuple)):
        raise ValueError("Solver did not return a list of indices")
    return [int(item) for item in path_index]


def route_distance(case: TSPCase, route: list[int]) -> float:
    node_ids = [case.depot.id] + route + [case.depot.id]
    if case.distance_matrix is not None:
        return float(
            sum(case.distance_matrix[current, nxt] for current, nxt in zip(node_ids, node_ids[1:]))
        )

    locations = {case.depot.id: case.depot, **{customer.id: customer for customer in case.customers}}
    distance = 0.0
    for current, nxt in zip(node_ids, node_ids[1:]):
        current_location = locations[current]
        next_location = locations[nxt]
        dx = next_location.x - current_location.x
        dy = next_location.y - current_location.y
        distance += math.sqrt(dx * dx + dy * dy)
    return float(distance)


def route_fuel(case: TSPCase, route: list[int]) -> float:
    locations = {case.depot.id: case.depot, **{customer.id: customer for customer in case.customers}}
    path = [locations[idx] for idx in route]
    remaining = sum(customer.weight for customer in path)
    capacity = 1.1 * remaining if remaining > 0 else 1.0
    fuel = 0.0
    previous = case.depot
    for current in path + [case.depot]:
        dx = current.x - previous.x
        dy = current.y - previous.y
        dist = math.sqrt(dx * dx + dy * dy)
        fuel += dist * (1.0 + (remaining / capacity))
        remaining -= current.weight
        previous = current
    return float(fuel)


def compute_gap(tour_length: float, optimal_length: float | None) -> float | None:
    if optimal_length is None or optimal_length <= 0 or not math.isfinite(optimal_length):
        return None
    return 100.0 * (tour_length - optimal_length) / optimal_length


def error_record(
        case: TSPCase,
        pool_name: str,
        status: str,
        error: str,
        runtime_seconds: float | None = None,
) -> dict:
    record = {
        "instance": case.name,
        "pool": pool_name,
        "dimension": len(case.customers) + 1,
        "source": case.source,
        "status": status,
        "tour_length": None,
        "reference_length": case.optimal_length,
        "reference_kind": "optimal" if case.optimal_length is not None else None,
        "gap": None,
        "error": error,
    }
    if runtime_seconds is not None:
        record["runtime_seconds"] = runtime_seconds
    return record


def summarize_records(records: list[dict], import_issues=None, timeout_seconds=None) -> dict:
    gaps = [record["gap"] for record in records if record.get("gap") is not None and math.isfinite(record["gap"])]
    lengths = [record["tour_length"] for record in records if record.get("tour_length") is not None]
    valid_count = sum(1 for record in records if record["status"] == "valid")
    timeout_count = sum(1 for record in records if record["status"] == "timeout")
    runtime_error_count = sum(1 for record in records if record["status"] == "error")
    invalid_count = sum(1 for record in records if record["status"] == "invalid")
    status = test_status(records)
    fitness = float(np.mean(gaps)) if gaps and status == "valid" else None
    return {
        "status": status,
        "fitness": fitness,
        "error_details": first_error(records),
        "metrics": {
            "pool": "test_instances",
            "runs": len(records),
            "valid_count": valid_count,
            "timeout_count": timeout_count,
            "runtime_error_count": runtime_error_count,
            "invalid_count": invalid_count,
            "timeout_seconds": timeout_seconds,
            "mean_gap": float(np.mean(gaps)) if gaps else None,
            "mean_tour_length": float(np.mean(lengths)) if lengths else None,
            "import_issues": import_issues or [],
            "records": records,
        },
    }


def test_status(records: list[dict]) -> str:
    if not records:
        return "invalid"
    if any(record["status"] == "timeout" for record in records):
        return "timeout"
    if any(record["status"] == "error" for record in records):
        return "error"
    if any(record["status"] == "invalid" for record in records):
        return "invalid"
    return "valid"


def first_nonvalid_status(records: list[dict]) -> str:
    for record in records:
        if record["status"] != "valid":
            return record["status"]
    return "valid"


def first_error(records: list[dict]) -> str | None:
    for record in records:
        if record["status"] != "valid":
            return f"{record['status']} on {record['instance']}: {record.get('error')}"
    return None


class evaluation_timeout:
    def __init__(self, timeout_seconds: float | None):
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
    raise TestEvaluationTimeoutError("evaluation timed out")


def select_candidate(solutions):
    if isinstance(solutions, ParetoArchive):
        solutions = solutions.get_best()
    if isinstance(solutions, Solution):
        return solutions
    return min(solutions, key=lambda solution: float(solution.fitness["Distance"]))


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Fitness):
        return value.to_dict()
    return value


if __name__ == "__main__":
    ai_model = os.getenv("LLM_MODEL", os.getenv("LLAMEA_LLM_MODEL", "gpt-5.4-nano"))
    search_seed = _int_env("LLAMEA_TSP_SEARCH_SEED", 69)
    search_size = _int_env("LLAMEA_TSP_SEARCH_SIZE", 32)
    llm_call_budget = _int_env("LLAMEA_LLM_CALLS", 27)
    n_parents = _int_env("LLAMEA_N_PARENTS", 3)
    n_offspring = _int_env("LLAMEA_N_OFFSPRING", 3)
    max_workers = _int_env("LLAMEA_MAX_WORKERS", 3)
    test_timeout = _float_env("LLAMEA_TSP_TEST_TIMEOUT", DEFAULT_TEST_TIMEOUT_SECONDS)
    experiment_name = os.getenv("LLAMEA_EXPERIMENT_NAME", "MOO-TSP")
    test_instances = os.getenv(
        "LLAMEA_TSP_TEST_INSTANCES",
        str(PROJECT_ROOT / "data" / "tsp" / "test_instances"),
    )

    _prepare_output_dir()
    SEARCH_CASE = build_generated_case(search_seed, search_size)
    test_cases = load_test_cases(test_instances)
    llm = _build_llm(ai_model)

    role_prompt = "You are an excellent Scientific Programmer, who can write novel solution to solve optimisation problem."

    task_prompt = """Write a novel solution, for solving multi-objective (Distance, Fuel) Travelling Salesman Problem.
The salesman starts and ends at the depot, and he visits each customer only once.
Write a class with __init__ method that excepts a two parameters.
    * The first one is the depot, which is of type tuple(int, int, int, int); corresponding to its id, x-coordinate, y-coordinate, weight.
    * The second is customers which is a `list[tuple(int, int, int, int)]`, same corresponding values for the tuple.
        * So the class should instantiate as `__init__(depot: tuple[int, int, int, int], customers: list[tuple[int, int, int, int]])`.
    * The class should also have a `__call__()` method, that returns the path as a list of customer ids: `list[int]`.
        * `Note`: The returned list must not contain depot's id, it is accounted for by the evaluator.
"""
    example_prompt = """
An example program of this solution will be:
import random
class Multi_Objective_TSP:
    def __init__(self, depot, customers):
        self.depot = depot
        self.customers = customers

    def __call__(self):
        customer_ids = [customer[0] for customer in self.customers]
        random.shuffle(customer_ids)
        return customer_ids
"""

    llamea_inst = LLaMEA(
        f=evaluate,
        llm=llm,
        multi_objective=True,
        max_workers=max_workers,
        n_offspring=n_offspring,
        n_parents=n_parents,
        multi_objective_keys=["Distance", "Fuel"],
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        example_prompt=example_prompt,
        experiment_name=experiment_name,
        minimization=True,
        budget=llm_call_budget,
    )

    solutions = llamea_inst.run()
    candidate = select_candidate(solutions)
    candidate.add_metadata("llm_model", llamea_inst.model)
    test_result = evaluate_test_solution(candidate, test_cases, timeout_seconds=test_timeout)
    payload = {
        "llm_model": llamea_inst.model,
        "search_instances": SEARCH_CASE.source,
        "test_instances": str(test_instances),
        "candidate": {
            "id": candidate.id,
            "name": candidate.name,
            "description": candidate.description,
            "search_fitness": candidate.fitness,
        },
        **test_result,
    }

    if getattr(llamea_inst, "logger", None) is not None:
        result_path = Path(llamea_inst.logger.dirname) / "test_result.json"
        result_path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        print(f"Wrote TSP test result: {result_path}")

    print(candidate.name)
    print(candidate.description)
    print(candidate.fitness)
    print(test_result)

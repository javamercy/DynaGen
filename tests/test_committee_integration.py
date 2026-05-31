"""Integration tests: committee mode end-to-end with mocked LLM, no API key needed."""

import os
from pathlib import Path

import pytest

from dynagen.candidates import ParsedCandidateResponse
from dynagen.config import RunConfig, _parse_simple_yaml
from dynagen.evolution.engine import EvolutionEngine
from dynagen.llm.base import LLMProvider, LLMResponse
from dynagen.persistence.run_store import RunStore
from dynagen.problems import problem_for_config

# ---------------------------------------------------------------------------
# Mock providers — one per problem type, each returns valid code
# ---------------------------------------------------------------------------


class _MockProvider(LLMProvider):
    def __init__(self, code: str, name: str = "mock_optimizer"):
        self._code = code
        self._name = name

    def complete(self, messages, *, temperature):
        return ParsedCandidateResponse(name=self._name, code=self._code, thought="mock")

    def complete_text(self, prompt, *, temperature):
        return ""

    def complete_with_metadata(self, messages, *, temperature):
        return LLMResponse(
            parsed_candidate_response=self.complete(messages, temperature=temperature),
            metadata={"model": "mock"},
        )


# Minimal valid code per problem type
BBOB_CODE = """\
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        best_x = np.zeros(self.dim)
        best_value = float("inf")
        for i in range(self.budget):
            x = self.rng.uniform(-5.0, 5.0, size=self.dim)
            val = func(x)
            if np.isfinite(val) and val < best_value:
                best_value = val
                best_x = x.copy()
            if i + 1 >= self.budget:
                func.report_best(best_value, best_x) if hasattr(func, "report_best") else None
        return best_x, best_value
"""

TSP_CODE = """\
import numpy as np

def solve_tsp(distance_matrix, seed, budget):
    n = len(distance_matrix)
    rng = np.random.default_rng(seed)
    best_tour = list(range(n))
    best_len = float("inf")
    for _ in range(min(budget, 10)):
        tour = list(rng.permutation(n))
        length = sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
        if length < best_len:
            best_len = length
            best_tour = tour
    return best_tour
"""

DVRP_CODE = """\
def choose_next_customer(current_location, available_customers, current_time,
                         time_windows, service_times, travel_times, truck_id,
                         truck_capacities, current_loads, customer_demands,
                         depot_location, max_capacity):
    if not available_customers:
        return None
    return available_customers[0]
"""

VRP_CODE = """\
def solve_vrp(customers, depot, vehicle_count, vehicle_capacity, seed, budget):
    routes = [[] for _ in range(vehicle_count)]
    for i, c in enumerate(customers):
        routes[i % vehicle_count].append(c)
    return routes
"""

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

BBOB_YAML = """\
run:
  name: test_bbob_committee
  output_dir: /tmp/dynagen_test
  seed: 1
problem:
  type: bbob
  function_ids: [ 1, 2 ]
  dimension: 2
  search_instances: [ 1 ]
  test_instances: [ 1 ]
  test_dimensions: [ 2 ]
  bounds: [ -5.0, 5.0 ]
  aocc_lower_bound: 0.00000001
  aocc_upper_bound: 100.0
llm:
  provider: deepseek
  model: mock
  temperature: 0.7
  api_key_env: DEEPSEEK_API_KEY
evolution:
  population_size: 2
  generations: 1
  offspring_per_strategy: 1
  output_mode: committee_specialist
  committee_size: 2
  strategies: [ m1_component_replacement ]
  history:
    enabled: true
    max_size: 10
    max_per_bucket: 2
  verbal_gradients:
    enabled: false
  niche:
    cadence_generations: 10
    improvement_weight: true
    improvement_power: 1.0
evaluation:
  budget: 100
  timeout_seconds: 30
  timeout_penalty: 0.0
  seeds: [ 1 ]
  metric: mean_aocc
"""

TSP_YAML = """\
run:
  name: test_tsp_committee
  output_dir: /tmp/dynagen_test
  seed: 1
problem:
  type: tsp
llm:
  provider: deepseek
  model: mock
  temperature: 0.7
  api_key_env: DEEPSEEK_API_KEY
evolution:
  population_size: 2
  generations: 1
  offspring_per_strategy: 1
  output_mode: committee_specialist
  committee_size: 2
  strategies: [ m1_component_replacement ]
  history:
    enabled: true
    max_size: 10
    max_per_bucket: 2
  verbal_gradients:
    enabled: false
  niche:
    cadence_generations: 10
    improvement_weight: true
    improvement_power: 1.0
evaluation:
  budget: 100
  timeout_seconds: 30
  timeout_penalty: 0.0
  seeds: [ 1 ]
  metric: average_gap
data:
  search_instances: synthetic:tsp_construct:n_instance=2:n_cities=10:seed=1
  test_instances: synthetic:tsp_construct:n_instance=2:n_cities=10:seed=2
"""

DVRP_YAML = """\
run:
  name: test_dvrp_committee
  output_dir: /tmp/dynagen_test
  seed: 1
problem:
  type: dvrp
  dvrp_search_limit: 2
  dvrp_test_sizes: [ 10 ]
  dvrp_test_limit_per_size: 2
llm:
  provider: deepseek
  model: mock
  temperature: 0.7
  api_key_env: DEEPSEEK_API_KEY
evolution:
  population_size: 2
  generations: 1
  offspring_per_strategy: 1
  output_mode: committee_specialist
  committee_size: 2
  strategies: [ m1_component_replacement ]
  history:
    enabled: true
    max_size: 10
    max_per_bucket: 2
  verbal_gradients:
    enabled: false
  niche:
    cadence_generations: 10
    improvement_weight: true
    improvement_power: 1.0
evaluation:
  timeout_seconds: 30
  timeout_penalty: 0.0
  seeds: [ 1 ]
  metric: mean_gap
data:
  search_instances: data/dvrp/train/instances.pkl
  test_instances: data/dvrp/test
"""

VRP_YAML = """\
run:
  name: test_vrp_committee
  output_dir: /tmp/dynagen_test
  seed: 1
problem:
  type: vrp
  vrp_search_limit: 2
  vrp_test_sizes: [ 10 ]
  vrp_test_limit_per_size: 2
llm:
  provider: deepseek
  model: mock
  temperature: 0.7
  api_key_env: DEEPSEEK_API_KEY
evolution:
  population_size: 2
  generations: 1
  offspring_per_strategy: 1
  output_mode: committee_specialist
  committee_size: 2
  strategies: [ m1_component_replacement ]
  history:
    enabled: true
    max_size: 10
    max_per_bucket: 2
  verbal_gradients:
    enabled: false
  niche:
    cadence_generations: 10
    improvement_weight: true
    improvement_power: 1.0
evaluation:
  timeout_seconds: 30
  timeout_penalty: 0.0
  metric: mean_gap
data:
  search_instances: data/vrp/train/instances.pkl
  test_instances: data/vrp/test
"""

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_engine(tmp_path, config, mock_provider):
    config.output_dir = str(tmp_path)
    problem = problem_for_config(config)
    search_eval = problem.build_evaluator(config, pool_name="search_instances")
    test_eval = problem.build_evaluator(config, pool_name="test_instances")
    store = RunStore.create(tmp_path, config.name, config.to_dict())
    engine = EvolutionEngine(
        config=config,
        provider=mock_provider,
        search_evaluator=search_eval,
        test_evaluator=test_eval,
        store=store,
    )
    population = engine.run()
    return population, store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bbob_committee_integration(tmp_path):
    config = RunConfig.from_dict(_parse_simple_yaml(BBOB_YAML))
    provider = _MockProvider(BBOB_CODE, "mock_bbob")
    population, store = _run_engine(tmp_path, config, provider)
    assert population is not None
    assert len(population.candidates) > 0
    assert Path(store.root, "final_report.md").exists()
    assert Path(store.root, "committee.json").exists()


def test_tsp_committee_integration(tmp_path):
    config = RunConfig.from_dict(_parse_simple_yaml(TSP_YAML))
    provider = _MockProvider(TSP_CODE, "mock_tsp")
    population, store = _run_engine(tmp_path, config, provider)
    assert population is not None
    assert len(population.candidates) > 0
    assert Path(store.root, "final_report.md").exists()
    assert Path(store.root, "committee.json").exists()


@pytest.mark.skipif(
    not (
        Path("data/dvrp/train/instances.pkl").exists()
        if Path.cwd().name == "DynaGen"
        else True
    ),
    reason="DVRP pickle data not found — run from project root with data/ present",
)
def test_dvrp_committee_integration(tmp_path):
    config = RunConfig.from_dict(_parse_simple_yaml(DVRP_YAML))
    provider = _MockProvider(DVRP_CODE, "mock_dvrp")
    population, store = _run_engine(tmp_path, config, provider)
    assert population is not None
    assert len(population.candidates) > 0
    assert Path(store.root, "final_report.md").exists()


@pytest.mark.skipif(
    not (
        Path("data/vrp/train/instances.pkl").exists()
        if Path.cwd().name == "DynaGen"
        else True
    ),
    reason="VRP pickle data not found — run from project root with data/ present",
)
def test_vrp_committee_integration(tmp_path):
    config = RunConfig.from_dict(_parse_simple_yaml(VRP_YAML))
    provider = _MockProvider(VRP_CODE, "mock_vrp")
    population, store = _run_engine(tmp_path, config, provider)
    assert population is not None
    assert len(population.candidates) > 0
    assert Path(store.root, "final_report.md").exists()

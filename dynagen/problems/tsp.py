from pathlib import Path
from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.config import RunConfig
from dynagen.domain import load_tsplib_file
from dynagen.domain.tsp_instance import TSPInstance
from dynagen.domain.tsp_synthetic import generate_llamea_tsp_instance, parse_llamea_tsp_spec
from dynagen.evaluation.tsp_evaluator import TSPCandidateEvaluator
from dynagen.prompts.tsp_evolution import build_tsp_evolution_prompt
from dynagen.prompts.tsp_initial import TSP_INITIAL_ROLES, TSPInitialRole, build_tsp_initial_prompt


class TSPProblem:
    type = "tsp"

    def build_evaluator(self, config: RunConfig, *, pool_name: str) -> TSPCandidateEvaluator:
        path = config.data.search_instances if pool_name == "search_instances" else config.data.test_instances
        return TSPCandidateEvaluator(
            load_tsp_instances(path),
            seeds=config.evaluation.seeds,
            budget=config.evaluation.budget,
            timeout_seconds=config.evaluation.timeout_seconds,
            timeout_penalty=config.evaluation.timeout_penalty,
            pool_name=pool_name,
        )

    def initial_roles(self, count: int) -> list[TSPInitialRole]:
        return create_tsp_initial_roles(count)

    def build_initial_prompt(self, role: Any) -> list[dict[str, str]]:
        return build_tsp_initial_prompt(role)

    def build_evolution_prompt(self, strategy: str, parents: list[Candidate]) -> list[dict[str, str]]:
        return build_tsp_evolution_prompt(strategy, parents)


def create_tsp_initial_roles(count: int) -> list[TSPInitialRole]:
    roles: list[TSPInitialRole] = []
    for index in range(count):
        role = TSP_INITIAL_ROLES[index % len(TSP_INITIAL_ROLES)]
        roles.append(TSPInitialRole(index + 1, role.role, role.intended_bias))
    return roles


def load_tsp_instances(path: str | Path | None) -> list[TSPInstance]:
    if not path:
        raise ValueError("TSP data.search_instances and data.test_instances must be specified")

    synthetic_spec = parse_llamea_tsp_spec(str(path))
    if synthetic_spec is not None:
        seed, size = synthetic_spec
        return [generate_llamea_tsp_instance(seed=seed, size=size)]

    path = Path(path)
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.suffix.lower() == ".tsp")
        if not files:
            raise ValueError(f"No .tsp files found in {path}")
        return [load_tsplib_file(file) for file in files]
    return [load_tsplib_file(path)]

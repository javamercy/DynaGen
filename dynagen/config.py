import json
import tomllib
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from dynagen.evolution.strategies import Strategy


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider must be specified")
        if not self.model:
            raise ValueError("model must be specified")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        if not self.provider.startswith("ollama") and not self.api_key_env:
            raise ValueError("api_key_env must be specified")


@dataclass
class VerbalGradientConfig:
    enabled: bool = True
    static_enabled: bool = True
    llm_enabled: bool = False
    llm_every_n_generations: int = 2
    max_llm_calls_per_generation: int = 2
    llm_model: str | None = None
    temperature: float = 0.2
    max_chars: int = 2000

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.static_enabled = bool(self.static_enabled)
        self.llm_enabled = bool(self.llm_enabled)
        self.llm_every_n_generations = int(self.llm_every_n_generations)
        self.max_llm_calls_per_generation = int(self.max_llm_calls_per_generation)
        if self.llm_model is not None:
            self.llm_model = str(self.llm_model).strip() or None
        self.temperature = float(self.temperature)
        self.max_chars = int(self.max_chars)
        if self.llm_every_n_generations < 1:
            raise ValueError("verbal_gradients.llm_every_n_generations must be at least 1")
        if self.max_llm_calls_per_generation < 0:
            raise ValueError("verbal_gradients.max_llm_calls_per_generation must be non-negative")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("verbal_gradients.temperature must be between 0 and 2")
        if self.max_chars < 200:
            raise ValueError("verbal_gradients.max_chars must be at least 200")


@dataclass
class EvolutionConfig:
    population_size: int
    generations: int
    offspring_per_strategy: int
    strategies: list[Strategy] = field(default_factory=lambda: list(Strategy))
    verbal_gradients: VerbalGradientConfig | dict[str, Any] = field(default_factory=VerbalGradientConfig)

    def __post_init__(self) -> None:
        self.population_size = int(self.population_size)
        self.generations = int(self.generations)
        self.offspring_per_strategy = int(self.offspring_per_strategy)
        self.strategies = [Strategy(strategy) for strategy in self.strategies]
        if isinstance(self.verbal_gradients, dict):
            self.verbal_gradients = VerbalGradientConfig(**self.verbal_gradients)
        elif not isinstance(self.verbal_gradients, VerbalGradientConfig):
            raise ValueError("evolution.verbal_gradients must be a mapping")
        if self.population_size < 1:
            raise ValueError("population_size must be at least 1")
        if self.generations < 0:
            raise ValueError("generations must be non-negative")
        if self.offspring_per_strategy < 0:
            raise ValueError("offspring_per_strategy must be non-negative")


@dataclass
class EvaluationConfig:
    budget: int
    timeout_seconds: float
    seeds: list[int]
    metric: str
    timeout_penalty: float = 10.0

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("At least one seed must be specified")

        if not self.metric:
            raise ValueError("metric must be specified")

        if not self.budget:
            raise ValueError("budget must be specified")

        if not self.timeout_seconds:
            raise ValueError("timeout_seconds must be specified")

        self.budget = int(self.budget)
        self.timeout_seconds = float(self.timeout_seconds)
        self.timeout_penalty = float(self.timeout_penalty)

        if self.budget < 1:
            raise ValueError("budget must be at least 1")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if self.timeout_penalty < 0:
            raise ValueError("timeout_penalty must be non-negative")


@dataclass
class ProblemConfig:
    type: str = "tsp"
    function_ids: list[int] = field(default_factory=lambda: list(range(1, 25)))
    dimension: int = 5
    search_instances: list[int] = field(default_factory=lambda: [1, 2, 3])
    test_instances: list[int] = field(default_factory=lambda: [1, 2, 3])
    test_dimensions: list[int] = field(default_factory=list)
    bounds: list[float] = field(default_factory=lambda: [-5.0, 5.0])
    aocc_lower_bound: float = 1e-8
    aocc_upper_bound: float = 1e2
    comparison_baselines: list[str] = field(default_factory=lambda: ["random_search", "differential_evolution"])
    dvrp_search_limit: int = 8
    dvrp_test_sizes: list[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    dvrp_test_limit_per_size: int = 64

    def __post_init__(self) -> None:
        self.type = str(self.type).lower()
        if self.type not in {"tsp", "bbob", "dvrp"}:
            raise ValueError("problem.type must be 'tsp', 'bbob', or 'dvrp'")

        self.function_ids = [int(function_id) for function_id in self.function_ids]
        self.dimension = int(self.dimension)
        self.search_instances = [int(instance_id) for instance_id in self.search_instances]
        self.test_instances = [int(instance_id) for instance_id in self.test_instances]
        self.test_dimensions = [int(dimension) for dimension in self.test_dimensions] or [self.dimension]
        self.bounds = [float(bound) for bound in self.bounds]
        self.aocc_lower_bound = float(self.aocc_lower_bound)
        self.aocc_upper_bound = float(self.aocc_upper_bound)
        self.comparison_baselines = [str(name) for name in self.comparison_baselines]
        self.dvrp_search_limit = int(self.dvrp_search_limit)
        self.dvrp_test_sizes = [int(size) for size in self.dvrp_test_sizes]
        self.dvrp_test_limit_per_size = int(self.dvrp_test_limit_per_size)

        if self.type == "bbob":
            _validate_bbob_problem_config(self)
        if self.type == "dvrp":
            _validate_dvrp_problem_config(self)


@dataclass
class DataConfig:
    search_instances: str | None = None
    test_instances: str | None = None

    def __post_init__(self) -> None:
        return None


def _validate_bbob_problem_config(config: ProblemConfig) -> None:
    if config.dimension < 1:
        raise ValueError("problem.dimension must be at least 1")
    if not config.function_ids:
        raise ValueError("problem.function_ids must not be empty")
    if any(function_id < 1 or function_id > 24 for function_id in config.function_ids):
        raise ValueError("BBOB function ids must be between 1 and 24")
    if not config.search_instances or not config.test_instances:
        raise ValueError("BBOB search_instances and test_instances must not be empty")
    if len(config.bounds) != 2 or config.bounds[0] >= config.bounds[1]:
        raise ValueError("problem.bounds must contain [lower, upper]")
    if config.aocc_lower_bound <= 0 or config.aocc_upper_bound <= config.aocc_lower_bound:
        raise ValueError("AOCC bounds must satisfy 0 < lower < upper")


def _validate_dvrp_problem_config(config: ProblemConfig) -> None:
    if config.dvrp_search_limit < 1:
        raise ValueError("problem.dvrp_search_limit must be at least 1")
    if not config.dvrp_test_sizes:
        raise ValueError("problem.dvrp_test_sizes must not be empty")
    if any(size < 2 for size in config.dvrp_test_sizes):
        raise ValueError("DVRP test sizes must be at least 2")
    if config.dvrp_test_limit_per_size < 1:
        raise ValueError("problem.dvrp_test_limit_per_size must be at least 1")


@dataclass
class RunConfig:
    name: str
    output_dir: str
    seed: int = 0
    llm: LLMConfig = field(default_factory=LLMConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        data = dict(data)
        run = dict(data.pop("run", {}))
        llm = LLMConfig(**data.pop("llm", {}))
        evolution = EvolutionConfig(**data.pop("evolution", {}))
        evaluation = EvaluationConfig(**data.pop("evaluation", {}))
        problem = ProblemConfig(**data.pop("problem", {}))
        data_cfg = DataConfig(**data.pop("data", {}))
        if data:
            raise ValueError(f"Unknown config keys: {sorted(data)}")

        return cls(
            name=run["name"],
            output_dir=run["output_dir"],
            seed=int(run.get("seed", 0)),
            llm=llm,
            evolution=evolution,
            evaluation=evaluation,
            problem=problem,
            data=data_cfg,
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


def load_config(path: str | Path | None = None) -> RunConfig:
    if path is None:
        raise ValueError("config path is required")
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
    elif path.suffix.lower() == ".toml":
        data = tomllib.loads(raw)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        data = _parse_simple_yaml(raw)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return RunConfig.from_dict(data)


def save_config(path: str | Path, config: RunConfig) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    value = value.split(" #", 1)[0].strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value

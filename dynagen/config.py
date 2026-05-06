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
class EvolutionConfig:
    population_size: int
    generations: int
    offspring_per_strategy: int
    strategies: list[Strategy] = field(default_factory=lambda: list(Strategy))

    def __post_init__(self) -> None:
        self.population_size = int(self.population_size)
        self.generations = int(self.generations)
        self.offspring_per_strategy = int(self.offspring_per_strategy)
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

        if self.budget < 1:
            raise ValueError("budget must be at least 1")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class DataConfig:
    search_instances: str
    test_instances: str

    def __post_init__(self) -> None:
        if not self.search_instances:
            raise ValueError("search_instances must be specified")
        if not self.test_instances:
            raise ValueError("test_instances must be specified")


@dataclass
class RunConfig:
    name: str
    output_dir: str
    seed: int = 0
    llm: LLMConfig = field(default_factory=LLMConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        data = dict(data)
        run = dict(data.pop("run", {}))
        llm = LLMConfig(**data.pop("llm", {}))
        evolution = EvolutionConfig(**data.pop("evolution", {}))
        evaluation = EvaluationConfig(**data.pop("evaluation", {}))
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

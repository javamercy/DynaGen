import argparse
import sys
from pathlib import Path

from dynagen.config import RunConfig, load_config
from dynagen.domain import load_tsplib_file
from dynagen.evaluation.evaluator import CandidateEvaluator
from dynagen.evolution.engine import EvolutionEngine
from dynagen.llm.ollama_provider import OllamaProvider
from dynagen.llm.openai_provider import OpenAIProvider
from dynagen.persistence.run_store import RunStore
from dynagen.reporting.summary import build_final_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dynagen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-run", help="Create a run directory with resolved config")
    init_parser.add_argument("--config", required=True, type=Path)

    run_parser = subparsers.add_parser("run", help="Run evolutionary TSP solver generation")
    run_parser.add_argument("--config", required=True, type=Path)

    eval_parser = subparsers.add_parser("evaluate-candidate",
                                        help="Evaluate a generated candidate on configured search data")
    eval_parser.add_argument("--candidate", required=True, type=Path)
    eval_parser.add_argument("--config", required=True, type=Path)

    summarize_parser = subparsers.add_parser("summarize", help="Print a run final report")
    summarize_parser.add_argument("--run", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "init-run":
        config = load_config(args.config)
        store = RunStore.create(config.output_dir, config.name, config.to_dict())
        print(store.root)
        return 0
    if args.command == "run":
        config = load_config(args.config)
        try:
            provider = _provider_from_config(config)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        search_instances = _load_instances(config.data.search_instances)
        search_evaluator = CandidateEvaluator(
            search_instances,
            seeds=config.evaluation.seeds,
            budget=config.evaluation.budget,
            timeout_seconds=config.evaluation.timeout_seconds,
            pool_name="search_instances",
        )
        test_instances = _load_instances(config.data.test_instances)
        test_evaluator = CandidateEvaluator(
            test_instances,
            seeds=config.evaluation.seeds,
            budget=config.evaluation.budget,
            timeout_seconds=config.evaluation.timeout_seconds,
            pool_name="test_instances",
        )

        store = RunStore.create(config.output_dir, config.name, config.to_dict())
        population = EvolutionEngine(
            config=config,
            provider=provider,
            search_evaluator=search_evaluator,
            test_evaluator=test_evaluator,
            store=store,
        ).run()
        print(store.root)
        print(f"best={population.best.id} fitness={population.best.fitness}")
        return 0
    if args.command == "evaluate-candidate":
        config = load_config(args.config)
        code = args.candidate.read_text(encoding="utf-8")
        instances = _load_instances(config.data.search_instances)
        search_evaluator = CandidateEvaluator(
            instances,
            seeds=config.evaluation.seeds,
            budget=config.evaluation.budget,
            timeout_seconds=config.evaluation.timeout_seconds,
            pool_name="search_instances",
        )
        result = search_evaluator.evaluate_code(code)
        print(build_final_report([]).splitlines()[0])
        print(f"status={result.status} fitness={result.fitness}")
        return 0
    if args.command == "summarize":
        report = args.run / "final_report.md"
        if report.exists():
            print(report.read_text(encoding="utf-8"))
        else:
            print(f"No final_report.md found in {args.run}")
            return 1
        return 0
    return 1


def _load_instances(path: str | Path):
    path = Path(path)
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.suffix.lower() == ".tsp")
        if not files:
            raise ValueError(f"No .tsp files found in {path}")
        return [load_tsplib_file(file) for file in files]
    return [load_tsplib_file(path)]


def _provider_from_config(config: RunConfig):
    if config.llm.provider == "openai":
        return OpenAIProvider(model=config.llm.model, api_key_env=config.llm.api_key_env)
    if config.llm.provider.startswith("ollama"):
        return OllamaProvider(model=config.llm.model)
    raise ValueError(f"Unsupported provider: {config.llm.provider}")


if __name__ == "__main__":
    raise SystemExit(main())

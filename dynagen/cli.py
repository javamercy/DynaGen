import argparse
import sys
from pathlib import Path

from dynagen.baselines.bbob import get_bbob_baseline_code
from dynagen.comparison.bbob import build_bbob_comparison_report, compare_bbob_candidate
from dynagen.config import RunConfig, load_config
from dynagen.domain import load_tsplib_file
from dynagen.domain.bbob import create_bbob_instances
from dynagen.evaluation.bbob_evaluator import BBOBCandidateEvaluator
from dynagen.evaluation.evaluator import CandidateEvaluator
from dynagen.evolution.engine import EvolutionEngine, scheduled_llm_calls
from dynagen.llm import CountingLLMProvider
from dynagen.persistence.run_store import RunStore
from dynagen.persistence.serialization import dump_json
from dynagen.reporting.summary import build_final_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dynagen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-run", help="Create a run directory with resolved config")
    init_parser.add_argument("--config", required=True, type=Path)

    run_parser = subparsers.add_parser("run", help="Run evolutionary solver/optimizer generation")
    run_parser.add_argument("--config", required=True, type=Path)

    eval_parser = subparsers.add_parser("evaluate-candidate",
                                        help="Evaluate a generated candidate on configured search data")
    eval_candidate_group = eval_parser.add_mutually_exclusive_group(required=True)
    eval_candidate_group.add_argument("--candidate", type=Path)
    eval_candidate_group.add_argument("--candidate-baseline")
    eval_parser.add_argument("--config", required=True, type=Path)

    summarize_parser = subparsers.add_parser("summarize", help="Print a run final report")
    summarize_parser.add_argument("--run", required=True, type=Path)

    compare_parser = subparsers.add_parser("compare-bbob", help="Compare a BBOB candidate against configured baselines")
    compare_candidate_group = compare_parser.add_mutually_exclusive_group()
    compare_candidate_group.add_argument("--candidate", type=Path)
    compare_candidate_group.add_argument("--candidate-baseline")
    compare_parser.add_argument("--config", required=True, type=Path)
    compare_parser.add_argument("--output", type=Path)

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
        try:
            search_evaluator = _build_evaluator(config, pool_name="search_instances")
            test_evaluator = _build_evaluator(config, pool_name="test_instances")
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

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
        try:
            code = _candidate_code_from_args(config, args)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        try:
            search_evaluator = _build_evaluator(config, pool_name="search_instances")
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        result = search_evaluator.evaluate_code(code)
        print(build_final_report([]).splitlines()[0])
        print(f"status={result.status} fitness={result.fitness}")
        return 0
    if args.command == "compare-bbob":
        config = load_config(args.config)
        if config.problem.type != "bbob":
            print("error: compare-bbob requires problem.type: bbob", file=sys.stderr)
            return 2
        try:
            code = _candidate_code_from_args(config, args, allow_none=True)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        try:
            if args.candidate_baseline:
                comparison = compare_bbob_candidate(
                    config,
                    code,
                    candidate_name=args.candidate_baseline,
                    candidate_kind="baseline",
                )
            elif args.candidate:
                comparison = compare_bbob_candidate(config, code, candidate_name=args.candidate.stem)
            else:
                comparison = compare_bbob_candidate(config)
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        report = build_bbob_comparison_report(comparison)
        if args.output:
            dump_json(args.output, comparison)
            report_path = args.output.with_suffix(".md")
            report_path.write_text(report, encoding="utf-8")
            print(args.output)
            print(report_path)
        else:
            print(report)
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


def _build_evaluator(config: RunConfig, *, pool_name: str):
    if config.problem.type == "bbob":
        instances = _load_bbob_instances(config, pool_name=pool_name)
        return BBOBCandidateEvaluator(
            instances,
            seeds=config.evaluation.seeds,
            budget=config.evaluation.budget,
            timeout_seconds=config.evaluation.timeout_seconds,
            timeout_penalty=config.evaluation.timeout_penalty,
            pool_name=pool_name,
            aocc_lower_bound=config.problem.aocc_lower_bound,
            aocc_upper_bound=config.problem.aocc_upper_bound,
        )
    instances = _load_instances(config.data.search_instances if pool_name == "search_instances" else config.data.test_instances)
    return CandidateEvaluator(
        instances,
        seeds=config.evaluation.seeds,
        budget=config.evaluation.budget,
        timeout_seconds=config.evaluation.timeout_seconds,
        timeout_penalty=config.evaluation.timeout_penalty,
        pool_name=pool_name,
    )


def _candidate_code_from_args(config: RunConfig, args, *, allow_none: bool = False) -> str | None:
    candidate_baseline = getattr(args, "candidate_baseline", None)
    if candidate_baseline:
        if config.problem.type != "bbob":
            raise ValueError("--candidate-baseline is only supported for problem.type: bbob")
        return get_bbob_baseline_code(candidate_baseline)
    candidate_path = getattr(args, "candidate", None)
    if candidate_path is None:
        if allow_none:
            return None
        raise ValueError("candidate is required")
    try:
        return candidate_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read candidate file {candidate_path}: {exc}") from exc


def _load_instances(path: str | Path | None):
    if not path:
        raise ValueError("TSP data.search_instances and data.test_instances must be specified")
    path = Path(path)
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.suffix.lower() == ".tsp")
        if not files:
            raise ValueError(f"No .tsp files found in {path}")
        return [load_tsplib_file(file) for file in files]
    return [load_tsplib_file(path)]


def _load_bbob_instances(config: RunConfig, *, pool_name: str):
    if pool_name == "search_instances":
        instance_ids = config.problem.search_instances
        dimensions = [config.problem.dimension]
    else:
        instance_ids = config.problem.test_instances
        dimensions = config.problem.test_dimensions
    return create_bbob_instances(
        function_ids=config.problem.function_ids,
        instance_ids=instance_ids,
        dimensions=dimensions,
        bounds=config.problem.bounds,
    )


def _provider_from_config(config: RunConfig):
    if config.llm.provider == "openai":
        from dynagen.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(model=config.llm.model, api_key_env=config.llm.api_key_env)
        return CountingLLMProvider(provider, configured_budget=scheduled_llm_calls(config))
    if config.llm.provider.startswith("ollama"):
        from dynagen.llm.ollama_provider import OllamaProvider

        provider = OllamaProvider(model=config.llm.model)
        return CountingLLMProvider(provider, configured_budget=scheduled_llm_calls(config))
    raise ValueError(f"Unsupported provider: {config.llm.provider}")


if __name__ == "__main__":
    raise SystemExit(main())

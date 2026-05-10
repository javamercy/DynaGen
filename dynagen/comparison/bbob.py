from pathlib import Path
from typing import Any

from dynagen.baselines.bbob import get_bbob_baseline_code
from dynagen.config import RunConfig
from dynagen.evaluation.base import CandidateEvaluator
from dynagen.problems import problem_for_config


def compare_bbob_candidate(
        config: RunConfig,
        candidate_code: str | None = None,
        *,
        candidate_name: str = "candidate",
        candidate_kind: str = "candidate",
) -> dict[str, Any]:
    if config.problem.type != "bbob":
        raise ValueError("BBOB comparison requires problem.type: bbob")
    evaluator = problem_for_config(config).build_evaluator(config, pool_name="bbob_comparison")
    algorithms = []
    evaluated_names: set[str] = set()
    if candidate_code is not None:
        algorithms.append(_evaluate_algorithm(evaluator, candidate_name, candidate_kind, candidate_code))
        evaluated_names.add(candidate_name)
    for baseline_name in config.problem.comparison_baselines:
        if baseline_name in evaluated_names:
            continue
        baseline_code = get_bbob_baseline_code(baseline_name)
        algorithms.append(_evaluate_algorithm(evaluator, baseline_name, "baseline", baseline_code))
        evaluated_names.add(baseline_name)

    algorithms.sort(key=lambda item: (item["fitness"] is None, float("inf") if item["fitness"] is None else item["fitness"]))
    return {
        "problem": "bbob",
        "settings": {
            "function_ids": config.problem.function_ids,
            "test_instances": config.problem.test_instances,
            "test_dimensions": config.problem.test_dimensions,
            "seeds": config.evaluation.seeds,
            "budget": config.evaluation.budget,
            "timeout_seconds": config.evaluation.timeout_seconds,
            "bounds": config.problem.bounds,
            "metric": "mean_aocc",
        },
        "algorithms": algorithms,
        "best_algorithm": algorithms[0]["name"] if algorithms else None,
    }


def build_bbob_comparison_report(comparison: dict[str, Any]) -> str:
    settings = comparison.get("settings", {})
    lines = [
        "# BBOB Comparison Report",
        "",
        "## Settings",
        "",
        f"- Functions: {settings.get('function_ids')}",
        f"- Test instances: {settings.get('test_instances')}",
        f"- Test dimensions: {settings.get('test_dimensions')}",
        f"- Seeds: {settings.get('seeds')}",
        f"- Budget: {settings.get('budget')}",
        f"- Bounds: {settings.get('bounds')}",
        "",
        "## Results",
        "",
        "| Rank | Algorithm | Kind | Status | Fitness | Mean AOCC | Mean Final Error | Valid Runs |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for rank, algorithm in enumerate(comparison.get("algorithms", []), start=1):
        metrics = algorithm.get("metrics", {})
        lines.append(
            f"| {rank} | {algorithm.get('name')} | {algorithm.get('kind')} | {algorithm.get('status')} | "
            f"{_fmt(algorithm.get('fitness'))} | {_fmt(metrics.get('mean_aocc'))} | "
            f"{_fmt(metrics.get('mean_final_error'))} | {metrics.get('valid_count')} / {metrics.get('runs')} |"
        )
    return "\n".join(lines) + "\n"


def write_bbob_comparison_report(path: str | Path, comparison: dict[str, Any]) -> None:
    Path(path).write_text(build_bbob_comparison_report(comparison), encoding="utf-8")


def _evaluate_algorithm(evaluator: CandidateEvaluator, name: str, kind: str, code: str) -> dict[str, Any]:
    result = evaluator.evaluate_code(code)
    return {
        "name": name,
        "kind": kind,
        "status": result.status,
        "fitness": result.fitness,
        "error_details": result.error_feedback,
        "metrics": result.metrics,
    }


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

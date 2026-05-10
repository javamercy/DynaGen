from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
from dynagen.evaluation.base import EvaluationResult
from dynagen.evolution.selection import select_survivors


def generation_summary(generation: int, population: list[Candidate], offspring: list[Candidate]) -> dict:
    best = select_survivors(population, 1)[0] if population else None
    valid_offspring = sum(1 for candidate in offspring if candidate.status == CandidateStatus.VALID)
    return {
        "generation": generation,
        "population_ids": [candidate.id for candidate in population],
        "offspring_count": len(offspring),
        "valid_offspring_count": valid_offspring,
        "best_candidate_id": None if best is None else best.id,
        "best_fitness": None if best is None else best.fitness,
        "status_counts": _status_counts(population + offspring),
    }


def build_final_report(
        population: list[Candidate],
        *,
        search_best: Candidate | None = None,
        test_result: EvaluationResult | None = None,
        llm_calls: dict | None = None,
) -> str:
    ordered = select_survivors(population, len(population)) if population else []
    lines = ["# DynaGen Final Report", "", "## Final Population", "",
             "| Rank | Candidate | Status | Search Fitness | Name |", "|---:|---|---|---:|---|"]
    for rank, candidate in enumerate(ordered, start=1):
        fitness = "" if candidate.fitness is None else f"{candidate.fitness:.6g}"
        lines.append(f"| {rank} | `{candidate.id}` | {candidate.status} | {fitness} | {candidate.name} |")
    if ordered:
        best = search_best or ordered[0]
        best_lines = [
            "",
            "## Search Best Candidate",
            "",
            f"- ID: `{best.id}`",
            f"- Name: {best.name}",
            f"- Status: {best.status}",
            f"- Search fitness: {best.fitness}",
            f"- Thought: {best.thought}",
        ]
        if best.error_details:
            best_lines.append(f"- Error details: {best.error_details}")
        lines.extend(best_lines)
    if test_result is not None:
        metrics = test_result.metrics
        runs = int(metrics.get("runs") or 0)
        seed_count = len(metrics.get("seeds") or [])
        instances_evaluated = runs // seed_count if seed_count else runs
        if metrics.get("problem") == "bbob":
            lines.extend(
                [
                    "",
                    "## Test Evaluation",
                    "",
                    f"- Problem: BBOB",
                    f"- Status: {test_result.status}",
                    f"- Test fitness: {test_result.fitness}",
                    f"- Problem instances evaluated: {instances_evaluated}",
                    f"- Valid runs: {metrics.get('valid_count')} / {metrics.get('runs')}",
                    f"- Scored runs: {metrics.get('scored_count')} / {metrics.get('runs')}",
                    f"- Partial timeout runs: {metrics.get('partial_timeout_count')}",
                    f"- Mean AOCC: {metrics.get('mean_aocc')}",
                    f"- Penalized mean AOCC: {metrics.get('penalized_mean_aocc')}",
                    f"- Median AOCC: {metrics.get('median_aocc')}",
                    f"- Best AOCC: {metrics.get('best_aocc')}",
                    f"- Worst AOCC: {metrics.get('worst_aocc')}",
                    f"- Mean final error: {metrics.get('mean_final_error')}",
                    f"- Best final error: {metrics.get('best_final_error')}",
                    f"- AOCC by group: {metrics.get('aocc_by_group')}",
                ]
            )
        else:
            search_metric_label = "Mean tour length" if metrics.get("pool") == "search_instances" else "Mean gap"
            penalized_metric_label = "Penalized mean tour length" if metrics.get("pool") == "search_instances" else "Penalized mean gap"
            lines.extend(
                [
                    "",
                    "## Test Evaluation",
                    "",
                    f"- Status: {test_result.status}",
                    f"- Test fitness: {test_result.fitness}",
                    f"- Instances evaluated: {instances_evaluated}",
                    f"- Valid runs: {metrics.get('valid_count')} / {metrics.get('runs')}",
                    f"- Scored runs: {metrics.get('scored_count')} / {metrics.get('runs')}",
                    f"- Partial timeout runs: {metrics.get('partial_timeout_count')}",
                    f"- {search_metric_label}: {metrics.get('mean_tour_length') if metrics.get('pool') == 'search_instances' else metrics.get('mean_gap')}",
                    f"- {penalized_metric_label}: {metrics.get('penalized_mean_gap') if metrics.get('pool') != 'search_instances' else metrics.get('mean_tour_length')}",
                    f"- Timeout penalty: {metrics.get('timeout_penalty')}",
                    f"- Median gap: {metrics.get('median_gap')}",
                    f"- Worst gap: {metrics.get('worst_gap')}",
                    f"- Best gap: {metrics.get('best_gap')}",
                ]
            )
        if test_result.error_feedback:
            lines.append(f"- Error details: {test_result.error_feedback}")
    if llm_calls is not None:
        lines.extend(
            [
                "",
                "## LLM Calls",
                "",
                f"- Candidate-generation calls: {llm_calls.get('candidate_generation_calls')}",
                f"- Total API calls: {llm_calls.get('total_api_calls')}",
                f"- Failed calls: {llm_calls.get('failed_calls')}",
                f"- LLM model: {llm_calls.get('llm_model') or llm_calls.get('model')}",
                f"- Configured candidate-generation budget: {llm_calls.get('configured_candidate_generation_budget')}",
                f"- Budget match: {llm_calls.get('budget_match')}",
            ]
        )
    return "\n".join(lines) + "\n"


def _status_counts(candidates: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.status] = counts.get(candidate.status, 0) + 1
    return counts

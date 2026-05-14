from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
from dynagen.evaluation.base import EvaluationResult
from dynagen.evolution.selection import select_survivors


def generation_summary(
        generation: int,
        population: list[Candidate],
        offspring: list[Candidate],
        *,
        archive_summary: dict | None = None,
) -> dict:
    best = select_survivors(population, 1)[0] if population else None
    valid_offspring = sum(1 for candidate in offspring if candidate.status == CandidateStatus.VALID)
    summary = {
        "generation": generation,
        "population_ids": [candidate.id for candidate in population],
        "offspring_count": len(offspring),
        "valid_offspring_count": valid_offspring,
        "best_candidate_id": None if best is None else best.id,
        "status_counts": _status_counts(population + offspring),
    }
    if best is not None:
        summary[f"best_{best.score_name}"] = best.score_value
    if archive_summary is not None:
        summary["archive"] = archive_summary
    return summary


def build_final_report(
        population: list[Candidate],
        *,
        search_best: Candidate | None = None,
        test_result: EvaluationResult | None = None,
        llm_calls: dict | None = None,
) -> str:
    ordered = select_survivors(population, len(population)) if population else []
    search_score_name = _population_score_name(ordered)
    search_score_title = _score_title(search_score_name)
    lines = ["# DynaGen Final Report", "", "## Final Population", "",
             f"| Rank | Candidate | Status | Search {search_score_title} | Name |", "|---:|---|---|---:|---|"]
    for rank, candidate in enumerate(ordered, start=1):
        score = "" if candidate.score_value is None else f"{candidate.score_value:.6g}"
        lines.append(f"| {rank} | `{candidate.id}` | {candidate.status} | {score} | {candidate.name} |")
    if ordered:
        best = search_best or ordered[0]
        best_lines = [
            "",
            "## Search Best Candidate",
            "",
            f"- ID: `{best.id}`",
            f"- Name: {best.name}",
            f"- Status: {best.status}",
            f"- Search {best.score_name}: {best.score_value}",
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
                    f"- Test fitness: {test_result.score}",
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
        elif metrics.get("problem") == "dvrp":
            lines.extend(
                [
                    "",
                    "## Test Evaluation",
                    "",
                    f"- Problem: DVRP",
                    f"- Status: {test_result.status}",
                    f"- Test distance: {test_result.score}",
                    f"- Instances evaluated: {instances_evaluated}",
                    f"- Valid runs: {metrics.get('valid_count')} / {metrics.get('runs')}",
                    f"- Scored runs: {metrics.get('scored_count')} / {metrics.get('runs')}",
                    f"- Mean gap: {metrics.get('mean_gap')}",
                    f"- Penalized mean gap: {metrics.get('penalized_mean_gap')}",
                    f"- Mean makespan: {metrics.get('mean_makespan')}",
                    f"- Timeout penalty: {metrics.get('timeout_penalty')}",
                    f"- Median gap: {metrics.get('median_gap')}",
                    f"- Worst gap: {metrics.get('worst_gap')}",
                    f"- Best gap: {metrics.get('best_gap')}",
                    f"- Gap by instance size: {metrics.get('score_by_instance_size')}",
                ]
            )
        else:
            search_metric_label = "Mean tour distance" if metrics.get("pool") == "search_instances" else "Mean gap"
            penalized_metric_label = "Penalized mean tour distance" if metrics.get("pool") == "search_instances" else "Penalized mean gap"
            lines.extend(
                [
                    "",
                    "## Test Evaluation",
                    "",
                    f"- Status: {test_result.status}",
                    f"- Test distance: {test_result.score}",
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
                f"- Reflection calls: {llm_calls.get('reflection_calls')}",
                f"- Feedback calls: {llm_calls.get('feedback_calls')}",
                f"- Total API calls: {llm_calls.get('total_api_calls')}",
                f"- Failed calls: {llm_calls.get('failed_calls')}",
                f"- Main LLM model: {llm_calls.get('llm_model') or llm_calls.get('model')}",
                f"- Feedback LLM model: {llm_calls.get('feedback_llm_model') or llm_calls.get('llm_model')}",
                f"- Configured candidate-generation budget: {llm_calls.get('configured_candidate_generation_budget')}",
                f"- Budget match: {llm_calls.get('budget_match')}",
            ]
        )
        verbal_gradients = llm_calls.get("verbal_gradients")
        if isinstance(verbal_gradients, dict):
            lines.extend([
                f"- Verbal gradients enabled: {verbal_gradients.get('enabled')}",
                f"- Verbal gradient cadence: every {verbal_gradients.get('llm_every_n_generations')} generations",
                f"- Verbal gradient model: {verbal_gradients.get('llm_model')}",
                f"- Static verbal gradients: {verbal_gradients.get('static_count')}",
                f"- LLM verbal gradients: {verbal_gradients.get('llm_count')}",
                f"- LLM verbal gradient errors: {verbal_gradients.get('llm_error_count')}",
            ])
        archive = llm_calls.get("archive")
        if isinstance(archive, dict):
            lines.extend([
                "",
                "## Archive",
                "",
                f"- Archive enabled: {archive.get('enabled')}",
                f"- Archive size: {archive.get('size')} / {archive.get('max_size')}",
                f"- Archive buckets: {archive.get('bucket_count')}",
                f"- Added candidates: {archive.get('added_count')}",
                f"- Duplicate rejections: {archive.get('rejected_duplicate_count')}",
                f"- Archive parent selections: {archive.get('parent_selections_from_archive')}",
                f"- Offspring with archive parent: {archive.get('offspring_with_archive_parent')}",
                f"- Final selection from archive: {bool(archive.get('final_selection_from_archive'))}",
            ])
    return "\n".join(lines) + "\n"


def _status_counts(candidates: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.status] = counts.get(candidate.status, 0) + 1
    return counts


def _population_score_name(candidates: list[Candidate]) -> str:
    for candidate in candidates:
        if candidate.score_name == "distance":
            return "distance"
    return "fitness"


def _score_title(score_name: str) -> str:
    return "Distance" if score_name == "distance" else "Fitness"

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
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
) -> str:
    ordered = select_survivors(population, len(population)) if population else []
    lines = ["# DynaGen Final Report", "", "## Final Population", "",
             "| Rank | Candidate | Status | Search Fitness | Name |", "|---:|---|---|---:|---|"]
    for rank, candidate in enumerate(ordered, start=1):
        fitness = "" if candidate.fitness is None else f"{candidate.fitness:.6g}"
        lines.append(f"| {rank} | `{candidate.id}` | {candidate.status} | {fitness} | {candidate.name} |")
    if ordered:
        best = search_best or ordered[0]
        lines.extend(
            [
                "",
                "## Search Best Candidate",
                "",
                f"- ID: `{best.id}`",
                f"- Name: {best.name}",
                f"- Status: {best.status}",
                f"- Search fitness: {best.fitness}",
                f"- Thought: {best.thought}",
            ]
        )
    return "\n".join(lines) + "\n"


def _status_counts(candidates: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.status] = counts.get(candidate.status, 0) + 1
    return counts

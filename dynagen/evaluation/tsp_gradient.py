from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import build_llm_gradient_messages


def build_tsp_llm_verbal_gradient_prompt(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
) -> list[dict[str, str]]:
    return build_llm_gradient_messages(
        problem="tsp",
        goal="minimize TSP tour length while always returning a valid permutation within the evaluator timeout",
        focus=(
            "construction heuristic, local-search neighborhood, restart/diversification behavior, "
            "report_best_tour use, timeout robustness, and size robustness"
        ),
        candidate=candidate,
        parents=parents,
        generation=generation,
    )

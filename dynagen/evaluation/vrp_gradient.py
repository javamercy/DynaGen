from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import build_llm_gradient_messages


def build_vrp_llm_verbal_gradient_prompt(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
) -> list[dict[str, str]]:
    return build_llm_gradient_messages(
        problem="vrp",
        goal="minimize the maximum route distance across all trucks while visiting every customer exactly once",
        focus=(
            "route construction, fleet balancing, split decisions, local search, restart behavior, "
            "route validity, report_best_vrp use, and robustness across customer counts"
        ),
        candidate=candidate,
        parents=parents,
        generation=generation,
    )

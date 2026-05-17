from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import build_llm_gradient_messages


def build_dvrp_llm_verbal_gradient_prompt(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
) -> list[dict[str, str]]:
    return build_llm_gradient_messages(
        problem="dvrp",
        goal="minimize the time until the last truck returns to the depot using only online dispatch state",
        focus=(
            "customer ranking, waiting behavior, fleet balance, depot-return pressure, spatial clustering, "
            "truck competition, per-call cost, and generalization across customer counts"
        ),
        candidate=candidate,
        parents=parents,
        generation=generation,
    )

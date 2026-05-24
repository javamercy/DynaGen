from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import build_llm_gradient_messages


def build_bbob_llm_verbal_gradient_prompt(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
) -> list[dict[str, str]]:
    return build_llm_gradient_messages(
        problem="bbob",
        goal="maximize mean AOCC under strict objective-evaluation budgets",
        focus=(
            "step-size control, covariance or coordinate adaptation, restart behavior, multimodal diversification, "
            "local refinement, bound handling, and budget accounting"
        ),
        candidate=candidate,
        parents=parents,
        generation=generation,
    )

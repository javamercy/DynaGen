from typing import Any, Protocol, TYPE_CHECKING

from dynagen.candidates.candidate import Candidate
from dynagen.evaluation.base import CandidateEvaluator

if TYPE_CHECKING:
    from dynagen.config import RunConfig


class Problem(Protocol):
    type: str

    def build_evaluator(self, config: "RunConfig", *, pool_name: str) -> CandidateEvaluator:
        ...

    def initial_roles(self, count: int) -> list[Any]:
        ...

    def build_initial_prompt(self, role: Any) -> list[dict[str, str]]:
        ...

    def build_evolution_prompt(
            self,
            strategy: str,
            parents: list[Candidate],
            *,
            feedback_context: str = "",
    ) -> list[dict[str, str]]:
        ...

    def build_static_verbal_gradient(
            self,
            candidate: Candidate,
            *,
            parents: list[Candidate],
            generation: int,
    ) -> dict[str, Any]:
        ...

    def build_llm_verbal_gradient_prompt(
            self,
            candidate: Candidate,
            *,
            parents: list[Candidate],
            generation: int,
            static_gradient: dict[str, Any],
    ) -> list[dict[str, str]]:
        ...

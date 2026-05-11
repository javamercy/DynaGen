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
            generation_reflection: str = "",
    ) -> list[dict[str, str]]:
        ...

    def build_llm_reflection_prompt(
            self,
            candidate: Candidate,
            *,
            parents: list[Candidate],
            generation: int,
    ) -> list[dict[str, str]]:
        ...

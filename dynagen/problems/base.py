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

    def build_llm_verbal_gradient_prompt(
            self,
            candidate: Candidate,
            *,
            parents: list[Candidate],
            generation: int,
    ) -> list[dict[str, str]]:
        ...

    def build_history_profile(self, candidate: Candidate) -> dict[str, Any]:
        ...

    def per_instance_scores(self, candidate: Candidate) -> dict[str, float]:
        """
        Return per-instance/function scores for this candidate.

        Returns:
            dict mapping instance key -> score value
            - BBOB: {"1": 0.92, "2": 0.86, ..., "24": 0.08}
            - TSP:   {"size:50": 0.12, ...} or {"instance:eil51": 0.15, ...}
            - DVRP:  {"size:10": 0.20, "trucks:3": 0.18, ...}
            - VRP:   {"size:10": 0.20, "trucks:3": 0.18, ...}
        """

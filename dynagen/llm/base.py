from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dynagen.candidates import ParsedCandidateResponse


@dataclass(frozen=True)
class LLMResponse:
    parsed_candidate_response: ParsedCandidateResponse
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[dict[str, str]], *, temperature: float) -> ParsedCandidateResponse:
        raise NotImplementedError

    @abstractmethod
    def complete_with_metadata(
            self,
            messages: list[dict[str, str]],
            *,
            temperature: float,
    ) -> LLMResponse:
        raise NotImplementedError


class CountingLLMProvider(LLMProvider):
    def __init__(self, provider: LLMProvider, *, configured_budget: int | None = None) -> None:
        self.provider = provider
        self.configured_budget = configured_budget
        self.candidate_generation_calls = 0
        self.total_api_calls = 0
        self.failed_calls = 0

    def complete(self, messages: list[dict[str, str]], *, temperature: float) -> ParsedCandidateResponse:
        self._record_call()
        try:
            return self.provider.complete(messages, temperature=temperature)
        except Exception:
            self.failed_calls += 1
            raise

    def complete_with_metadata(
            self,
            messages: list[dict[str, str]],
            *,
            temperature: float,
    ) -> LLMResponse:
        self._record_call()
        try:
            return self.provider.complete_with_metadata(messages, temperature=temperature)
        except Exception:
            self.failed_calls += 1
            raise

    def summary(self) -> dict[str, Any]:
        budget_match = None
        if self.configured_budget is not None:
            budget_match = self.candidate_generation_calls == self.configured_budget
        return {
            "candidate_generation_calls": self.candidate_generation_calls,
            "total_api_calls": self.total_api_calls,
            "failed_calls": self.failed_calls,
            "configured_candidate_generation_budget": self.configured_budget,
            "budget_match": budget_match,
        }

    def _record_call(self) -> None:
        self.candidate_generation_calls += 1
        self.total_api_calls += 1

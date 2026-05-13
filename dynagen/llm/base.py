import threading
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

    @abstractmethod
    def complete_text(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        raise NotImplementedError


class CountingLLMProvider(LLMProvider):
    def __init__(self, provider: LLMProvider, *, configured_budget: int | None = None) -> None:
        self.provider = provider
        self.configured_budget = configured_budget
        self._candidate_generation_calls = 0
        self._feedback_calls = 0
        self._total_api_calls = 0
        self._failed_calls = 0
        self._lock = threading.Lock()

    @property
    def candidate_generation_calls(self) -> int:
        with self._lock:
            return self._candidate_generation_calls

    @property
    def total_api_calls(self) -> int:
        with self._lock:
            return self._total_api_calls

    @property
    def failed_calls(self) -> int:
        with self._lock:
            return self._failed_calls

    @property
    def reflection_calls(self) -> int:
        return self.feedback_calls

    @property
    def feedback_calls(self) -> int:
        with self._lock:
            return self._feedback_calls

    def complete(self, messages: list[dict[str, str]], *, temperature: float) -> ParsedCandidateResponse:
        self._record_call()
        try:
            return self.provider.complete(messages, temperature=temperature)
        except Exception:
            self._record_failure()
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
            self._record_failure()
            raise

    def complete_text(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        self._record_feedback_call()
        try:
            return self.provider.complete_text(messages, temperature=temperature)
        except Exception:
            self._record_failure()
            raise

    def summary(self) -> dict[str, Any]:
        with self._lock:
            calls = self._candidate_generation_calls
            feedback = self._feedback_calls
            total = self._total_api_calls
            failed = self._failed_calls
        budget_match = None
        if self.configured_budget is not None:
            budget_match = calls == self.configured_budget
        return {
            "candidate_generation_calls": calls,
            "reflection_calls": feedback,
            "feedback_calls": feedback,
            "total_api_calls": total,
            "failed_calls": failed,
            "configured_candidate_generation_budget": self.configured_budget,
            "budget_match": budget_match,
        }

    def _record_call(self) -> None:
        with self._lock:
            self._candidate_generation_calls += 1
            self._total_api_calls += 1

    def _record_feedback_call(self) -> None:
        with self._lock:
            self._feedback_calls += 1
            self._total_api_calls += 1

    def _record_failure(self) -> None:
        with self._lock:
            self._failed_calls += 1

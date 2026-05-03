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

# DONE

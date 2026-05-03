import os
from typing import Any
from openai import OpenAI

from dynagen.candidates import ParsedCandidateResponse
from dynagen.candidates.parser import CANDIDATE_RESPONSE_SCHEMA
from dynagen.llm.base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    def __init__(self, *, model: str, api_key_env: str = "OPENAI_API_KEY") -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenAI API key environment variable: {api_key_env}. "
                f"Set it in the same shell with: export {api_key_env}=<your_api_key>"
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, messages: list[dict[str, str]], *, temperature: float) -> ParsedCandidateResponse:
        return self.complete_with_metadata(messages, temperature=temperature).parsed_candidate_response

    def complete_with_metadata(self, messages: list[dict[str, str]], *, temperature: float) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "candidate_response",
                    "strict": True,
                    "schema": CANDIDATE_RESPONSE_SCHEMA,
                },
            }
        )
        content = response.choices[0].message.content or ""
        metadata: dict[str, Any] = {"model": self.model}
        usage = getattr(response, "usage", None)
        if usage is not None:
            metadata["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)

        return LLMResponse(
            parsed_candidate_response=ParsedCandidateResponse.from_json(content),
            metadata=metadata,
        )

# DONE

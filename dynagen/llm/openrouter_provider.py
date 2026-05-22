import os
from typing import Any
from openai import OpenAI

from dynagen.candidates import ParsedCandidateResponse
from dynagen.candidates.parser import CANDIDATE_RESPONSE_SCHEMA
from dynagen.llm.base import LLMProvider, LLMResponse


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider using OpenAI-compatible API.
    
    OpenRouter's API is fully compatible with OpenAI's Python client.
    This provider uses the OpenAI client with OpenRouter's endpoint.
    """
    
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str | None = None,
    ) -> None:
        """Initialize OpenRouter provider.
        
        Args:
            model: Model name (e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet")
            api_key_env: Environment variable containing API key
            base_url: Optional custom API endpoint (default: OpenRouter official API)
        """
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenRouter API key environment variable: {api_key_env}. "
                f"Set it in the same shell with: export {api_key_env}=<your_api_key>"
            )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or self.OPENROUTER_API_BASE,
        )
        self.model = model

    def complete(self, messages: list[dict[str, str]], *, temperature: float) -> ParsedCandidateResponse:
        """Generate candidate response."""
        return self.complete_with_metadata(messages, temperature=temperature).parsed_candidate_response

    def complete_text(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        """Generate plain text response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def complete_with_metadata(self, messages: list[dict[str, str]], *, temperature: float) -> LLMResponse:
        """Generate response with metadata (JSON format)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_object",
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

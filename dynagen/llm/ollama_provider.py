import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dynagen.candidates import ParsedCandidateResponse
from dynagen.candidates.parser import CANDIDATE_RESPONSE_SCHEMA
from dynagen.llm.base import LLMProvider, LLMResponse

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT_SECONDS = 120
OLLAMA_METADATA_KEYS = (
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


class OllamaProvider(LLMProvider):
    def __init__(self, *, model: str) -> None:
        self.model = model

    def complete(self, messages: list[dict[str, str]], *, temperature: float) -> ParsedCandidateResponse:
        return self.complete_with_metadata(messages, temperature=temperature).parsed_candidate_response

    def complete_text(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        return self._message_content(self._chat(messages, temperature=temperature, format_schema=None))

    def complete_with_metadata(self, messages: list[dict[str, str]], *, temperature: float) -> LLMResponse:
        response_data = self._chat(messages, temperature=temperature)
        content = self._message_content(response_data)
        print(f"Ollama response content: {content}")
        return LLMResponse(
            parsed_candidate_response=ParsedCandidateResponse.from_json(content),
            metadata=self._metadata(response_data),
        )

    def _chat(
            self,
            messages: list[dict[str, str]],
            *,
            temperature: float,
            format_schema: dict[str, Any] | None = CANDIDATE_RESPONSE_SCHEMA,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if format_schema is not None:
            payload["format"] = format_schema
        request = Request(
            OLLAMA_CHAT_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed: HTTP {exc.code}: {body}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama request timed out after {OLLAMA_TIMEOUT_SECONDS}s for model {self.model}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {OLLAMA_CHAT_URL}. "
                "Make sure Ollama is running locally."
            ) from exc
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError("Ollama response must be a JSON object")
        return data

    @staticmethod
    def _message_content(response_data: dict[str, Any]) -> str:
        message = response_data.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        return content if isinstance(content, str) else ""

    def _metadata(self, response_data: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {"model": self.model}
        for key in OLLAMA_METADATA_KEYS:
            if key in response_data:
                metadata[key] = response_data[key]
        return metadata

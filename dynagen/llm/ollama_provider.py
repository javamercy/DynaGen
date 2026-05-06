import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dynagen.candidates import ParsedCandidateResponse
from dynagen.candidates.parser import CANDIDATE_RESPONSE_SCHEMA
from dynagen.llm.base import LLMProvider, LLMResponse
from dynagen.prompts.templates import RESPONSE_FORMAT


class OllamaProvider(LLMProvider):
    def __init__(self, *, model: str) -> None:
        self.model = model
        self.is_cloud = model.endswith(":cloud")

    def complete(self, messages: list[dict[str, str]], *, temperature: float) -> ParsedCandidateResponse:
        return self.complete_with_metadata(messages, temperature=temperature).parsed_candidate_response

    def complete_with_metadata(self, messages: list[dict[str, str]], *, temperature: float) -> LLMResponse:
        if self.is_cloud:
            return self._complete_cloud(messages, temperature=temperature)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": CANDIDATE_RESPONSE_SCHEMA,
            "options": {
                "temperature": temperature,
            },
        }
        request = Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed: HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(
                "Unable to reach Ollama at http://localhost:11434. "
                "Make sure Ollama is running locally."
            ) from exc

        response_data = json.loads(raw)
        content = ((response_data.get("message") or {}).get("content")) or ""
        metadata: dict[str, Any] = {"model": self.model}
        for key in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count",
                    "eval_duration"):
            if key in response_data:
                metadata[key] = response_data[key]

        return LLMResponse(
            parsed_candidate_response=ParsedCandidateResponse.from_json(content),
            metadata=metadata,
        )

    def _complete_cloud(self, messages: list[dict[str, str]], *, temperature: float) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object with keys name, thought, and code. "
                        "Do not use markdown, code fences, or extra text. "
                        f"Schema: {json.dumps(CANDIDATE_RESPONSE_SCHEMA, separators=(',', ':'))}\n\n"
                        f"{RESPONSE_FORMAT}"
                    ),
                },
                *messages,
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        request = Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            print("Ollama request failed:", exc)
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed: HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(
                "Unable to reach Ollama at http://localhost:11434. "
                "Make sure Ollama is running locally and the cloud model is available."
            ) from exc

        response_data = json.loads(raw)
        content = ((response_data.get("message") or {}).get("content")) or ""
        metadata: dict[str, Any] = {"model": self.model, "cloud": True}
        for key in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count",
                    "eval_duration"):
            if key in response_data:
                metadata[key] = response_data[key]

        return LLMResponse(
            parsed_candidate_response=_parse_candidate_response(content),
            metadata=metadata,
        )


def _parse_candidate_response(content: str) -> ParsedCandidateResponse:
    content = content.strip()
    try:
        return ParsedCandidateResponse.from_json(content)
    except Exception:
        pass

    if content.startswith("```"):
        content = _strip_code_fences(content)

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = content[start: end + 1]
        return ParsedCandidateResponse.from_json(candidate)

    raise ValueError("Candidate response must be a JSON object")


def _strip_code_fences(content: str) -> str:
    lines = content.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

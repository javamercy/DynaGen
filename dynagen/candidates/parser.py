import json
from dataclasses import dataclass
from typing import Any

CANDIDATE_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "thought", "code"],
    "properties": {
        "name": {
            "type": "string",
            "minLength": 5,
            "maxLength": 25,
            "description": "A short descriptive name for the generated TSP solver.",
        },
        "thought": {
            "type": "string",
            "minLength": 10,
            "maxLength": 500,
            "description": "A brief high-level explanation of the solver strategy.",
        },
        "code": {
            "type": "string",
            "minLength": 1,
            "description": "Complete Python source code defining solve_tsp(distance_matrix, seed, budget).",
        },
    },
}


@dataclass(frozen=True)
class ParsedCandidateResponse:
    name: str
    thought: str
    code: str

    @classmethod
    def from_json(cls, response: str) -> "ParsedCandidateResponse":
        data: Any = json.loads(response)

        if not isinstance(data, dict):
            raise ValueError("Candidate response must be a JSON object")

        missing = {"name", "thought", "code"} - data.keys()
        if missing:
            raise ValueError(f"Candidate response is missing required fields: {sorted(missing)}")

        return cls(
            name=str(data["name"]).strip(),
            thought=str(data["thought"]).strip(),
            code=str(data["code"]).strip(),
        )

# DONE

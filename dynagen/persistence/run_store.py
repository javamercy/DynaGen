from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.evaluation.evaluator import EvaluationResult
from dynagen.persistence.serialization import dump_json, load_json


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.generations_dir = self.root / "generations"
        self.candidates_dir = self.root / "candidates"
        self.prompts_dir = self.root / "prompts"
        self._counter = self._scan_counter()
        for directory in (self.generations_dir, self.candidates_dir, self.prompts_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, base_dir: str | Path, run_name: str, config: dict[str, Any]) -> "RunStore":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in run_name).strip("_")
        base_root = Path(base_dir) / f"{timestamp}_{safe_name or 'run'}"
        root = base_root
        suffix = 1
        while root.exists():
            root = Path(f"{base_root}_{suffix:02d}")
            suffix += 1
        store = cls(root)
        dump_json(store.root / "config.json", config)
        return store

    def next_candidate_id(self) -> str:
        self._counter += 1
        return f"cand_{self._counter:06d}"

    def save_candidate(self, candidate: Candidate) -> None:
        dump_json(self.candidates_dir / f"{candidate.id}.json", candidate.to_dict(include_code=False))
        (self.candidates_dir / f"{candidate.id}.py").write_text(candidate.code, encoding="utf-8")
        if candidate.prompt:
            (self.prompts_dir / f"{candidate.id}_prompt.txt").write_text(candidate.prompt, encoding="utf-8")
        if candidate.raw_response:
            (self.prompts_dir / f"{candidate.id}_response.txt").write_text(candidate.raw_response, encoding="utf-8")

    def load_candidate(self, candidate_id: str) -> Candidate:
        metadata = load_json(self.candidates_dir / f"{candidate_id}.json")
        code_path = self.candidates_dir / f"{candidate_id}.py"
        code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
        return Candidate.from_dict(metadata, code=code)

    def save_generation(
            self,
            generation: int,
            *,
            population: list[Candidate],
            offspring: list[Candidate] | None = None,
            summary: dict[str, Any] | None = None,
    ) -> None:
        generation_dir = self.generations_dir / f"generation_{generation:03d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        dump_json(generation_dir / "population.json",
                  [candidate.to_dict(include_code=False) for candidate in population])
        dump_json(generation_dir / "offspring.json",
                  [candidate.to_dict(include_code=False) for candidate in offspring or []])
        dump_json(generation_dir / "summary.json", summary or {})

    def save_split_manifest(self, manifest: dict[str, Any]) -> None:
        dump_json(self.root / "split_manifest.json", manifest)

    def save_test_result(self, candidate_id: str, result: EvaluationResult) -> None:
        dump_json(self.root / "test_result.json", {
            "candidate_id": candidate_id,
            "status": result.status,
            "fitness": result.fitness,
            "error_details": result.error_feedback,
            "metrics": result.metrics,
        })

    def save_llm_calls(self, summary: dict[str, Any]) -> None:
        dump_json(self.root / "llm_calls.json", summary)

    def write_final_report(self, text: str) -> None:
        (self.root / "final_report.md").write_text(text, encoding="utf-8")

    def _scan_counter(self) -> int:
        if not self.candidates_dir.exists():
            return 0
        max_id = 0
        for path in self.candidates_dir.glob("cand_*.json"):
            try:
                max_id = max(max_id, int(path.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max_id

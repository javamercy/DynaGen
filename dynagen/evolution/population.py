from __future__ import annotations

from dataclasses import dataclass

from dynagen.candidates.candidate import Candidate
from dynagen.evolution.selection import select_survivors


@dataclass
class Population:
    generation: int
    candidates: list[Candidate]

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("Generation must be non-negative")
        if not self.candidates:
            raise ValueError("Candidates list cannot be empty")

    @classmethod
    def from_candidates(cls, generation: int, candidates: list[Candidate], *, size: int) -> "Population":
        if size <= 0:
            raise ValueError("Size must be positive")

        return cls(generation=generation, candidates=select_survivors(candidates, size))

    @property
    def best(self) -> Candidate:
        return select_survivors(self.candidates, 1)[0]

    def ids(self) -> list[str]:
        return [candidate.id for candidate in self.candidates]

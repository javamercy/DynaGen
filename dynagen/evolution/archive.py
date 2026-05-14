from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.config import ArchiveConfig

ARCHIVE_KEY = "archive"
ARCHIVE_SELECTION_KEY = "archive_selection"


@dataclass
class ArchiveEntry:
    candidate_id: str
    problem: str
    generation: int
    status: str
    score_name: str
    score_value: float | None
    buckets: list[str]
    primary_bucket: str
    archive_score: float
    quality_score: float
    robustness_score: float
    diversity_score: float
    recency_score: float
    code_hash: str | None
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    bucket_scores: dict[str, float] = field(default_factory=dict)
    diversity_features: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "problem": self.problem,
            "generation": self.generation,
            "status": self.status,
            "score_name": self.score_name,
            "score_value": self.score_value,
            "buckets": list(self.buckets),
            "primary_bucket": self.primary_bucket,
            "archive_score": self.archive_score,
            "quality_score": self.quality_score,
            "robustness_score": self.robustness_score,
            "diversity_score": self.diversity_score,
            "recency_score": self.recency_score,
            "code_hash": self.code_hash,
            "metrics_snapshot": self.metrics_snapshot,
            "bucket_scores": self.bucket_scores,
            "diversity_features": self.diversity_features,
            "created_at": self.created_at,
        }


class CandidateArchive:
    def __init__(self, *, config: ArchiveConfig, problem: str) -> None:
        self.config = config
        self.problem = problem
        self.entries: dict[str, ArchiveEntry] = {}
        self.stats: dict[str, int] = {
            "added_count": 0,
            "updated_count": 0,
            "rejected_status_count": 0,
            "rejected_score_count": 0,
            "rejected_duplicate_count": 0,
            "pruned_count": 0,
            "parent_selections_from_archive": 0,
            "offspring_with_archive_parent": 0,
            "final_selection_from_archive": 0,
        }

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def update(
            self,
            candidates: list[Candidate],
            *,
            generation: int,
            profile_builder,
    ) -> None:
        if not self.enabled:
            return
        for candidate in candidates:
            self.add_candidate(candidate, generation=generation, profile_builder=profile_builder)
        self._recompute_scores(current_generation=generation)
        self._prune()

    def add_candidate(self, candidate: Candidate, *, generation: int, profile_builder) -> ArchiveEntry | None:
        status = _status_value(candidate)
        if status not in set(self.config.add_statuses):
            self.stats["rejected_status_count"] += 1
            _set_archive_metadata(candidate, in_archive=False, reason="status")
            return None

        score_value = _finite_or_none(candidate.score_value)
        if score_value is None:
            self.stats["rejected_score_count"] += 1
            _set_archive_metadata(candidate, in_archive=False, reason="score")
            return None

        profile = dict(profile_builder(candidate))
        buckets = _clean_buckets(profile.get("buckets"))
        if not buckets:
            buckets = ["global"]
        primary_bucket = str(profile.get("primary_bucket") or buckets[0])
        if primary_bucket not in buckets:
            buckets.insert(0, primary_bucket)

        code_hash = normalized_code_hash(candidate.code) if self.config.deduplicate_code else None
        if code_hash:
            duplicate = self._duplicate_for_hash(code_hash, exclude_id=candidate.id)
            if duplicate is not None:
                new_quality = _score(profile.get("quality_score"))
                if duplicate.quality_score >= new_quality:
                    self.stats["rejected_duplicate_count"] += 1
                    _set_archive_metadata(candidate, in_archive=False, reason="duplicate")
                    return None
                self._remove_entry(duplicate.candidate_id)

        existing = self.entries.get(candidate.id)
        diversity_features = _dict(profile.get("diversity_features"))
        diversity_score = _score(profile.get("diversity_score"))
        if "diversity_score" not in profile:
            diversity_score = self._diversity_score(primary_bucket, diversity_features, exclude_id=candidate.id)
        entry = ArchiveEntry(
            candidate_id=candidate.id,
            problem=self.problem,
            generation=int(candidate.generation),
            status=status,
            score_name=candidate.score_name,
            score_value=score_value,
            buckets=buckets,
            primary_bucket=primary_bucket,
            archive_score=0.0,
            quality_score=_score(profile.get("quality_score")),
            robustness_score=_score(profile.get("robustness_score")),
            diversity_score=diversity_score,
            recency_score=0.0,
            code_hash=code_hash,
            metrics_snapshot=_dict(profile.get("metrics_snapshot")),
            bucket_scores=_score_dict(profile.get("bucket_scores")),
            diversity_features=diversity_features,
        )
        self.entries[candidate.id] = entry
        self._recompute_entry_score(entry, current_generation=generation)
        self.stats["updated_count" if existing else "added_count"] += 1
        _set_archive_metadata(candidate, in_archive=True, entry=entry)
        return entry

    def select_parents(
            self,
            *,
            count: int,
            rng: random.Random,
            candidate_index: dict[str, Candidate],
            exclude_ids: set[str] | None = None,
            diversify_buckets: bool = False,
    ) -> list[Candidate]:
        selected: list[Candidate] = []
        used_ids = set(exclude_ids or set())
        used_buckets: set[str] = set()
        for _ in range(max(0, count)):
            choices = self._available_parent_entries(candidate_index, exclude_ids=used_ids)
            if diversify_buckets:
                diverse_choices = [
                    item for item in choices
                    if item[0].primary_bucket not in used_buckets
                ]
                if diverse_choices:
                    choices = diverse_choices
            if not choices:
                break
            entry, candidate = _rank_biased_archive_choice(choices, rng)
            selected.append(candidate)
            used_ids.add(candidate.id)
            used_buckets.add(entry.primary_bucket)
            _set_archive_selection(candidate, entry)
        if selected:
            self.stats["parent_selections_from_archive"] += len(selected)
        return selected

    def candidates(self, candidate_index: dict[str, Candidate]) -> list[Candidate]:
        result: list[Candidate] = []
        for entry in sorted(self.entries.values(), key=_entry_sort_key):
            candidate = candidate_index.get(entry.candidate_id)
            if candidate is not None:
                result.append(candidate)
        return result

    def candidate_ids(self) -> set[str]:
        return set(self.entries)

    def note_offspring_with_archive_parent(self) -> None:
        self.stats["offspring_with_archive_parent"] += 1

    def mark_final_selection(self, candidate_id: str, *, population_ids: set[str] | None = None) -> None:
        population_ids = set(population_ids or set())
        self.stats["final_selection_from_archive"] = int(candidate_id in self.entries and candidate_id not in population_ids)

    def summary(self, *, include_entries: bool = True) -> dict[str, Any]:
        bucket_map = self._bucket_map()
        top_buckets = []
        for bucket in sorted(bucket_map):
            entries = sorted(bucket_map[bucket], key=lambda entry: _bucket_sort_key(entry, bucket))
            if entries:
                top_buckets.append({
                    "bucket": bucket,
                    "candidate_id": entries[0].candidate_id,
                    "archive_score": entries[0].archive_score,
                    "bucket_score": entries[0].bucket_scores.get(bucket, entries[0].archive_score),
                })
        summary = {
            "enabled": self.enabled,
            "size": len(self.entries),
            "max_size": self.config.max_size,
            "max_per_bucket": self.config.max_per_bucket,
            "bucket_count": len(bucket_map),
            **self.stats,
            "top_buckets": top_buckets[:20],
        }
        if include_entries:
            summary["entries"] = [entry.to_dict() for entry in sorted(self.entries.values(), key=_entry_sort_key)]
        return summary

    def _available_parent_entries(
            self,
            candidate_index: dict[str, Candidate],
            *,
            exclude_ids: set[str],
    ) -> list[tuple[ArchiveEntry, Candidate]]:
        entries: list[tuple[ArchiveEntry, Candidate]] = []
        for entry in self.entries.values():
            if entry.candidate_id in exclude_ids:
                continue
            candidate = candidate_index.get(entry.candidate_id)
            if candidate is None:
                continue
            entries.append((entry, candidate))
        return sorted(entries, key=lambda item: _entry_sort_key(item[0]))

    def _duplicate_for_hash(self, code_hash: str, *, exclude_id: str) -> ArchiveEntry | None:
        for entry in self.entries.values():
            if entry.candidate_id != exclude_id and entry.code_hash == code_hash:
                return entry
        return None

    def _diversity_score(
            self,
            primary_bucket: str,
            diversity_features: dict[str, Any],
            *,
            exclude_id: str,
    ) -> float:
        same_bucket = [
            entry for entry in self.entries.values()
            if entry.candidate_id != exclude_id and entry.primary_bucket == primary_bucket
        ]
        if not same_bucket:
            return 1.0
        if not diversity_features:
            return 0.5
        if all(entry.diversity_features != diversity_features for entry in same_bucket):
            return 1.0
        return 0.2

    def _recompute_scores(self, *, current_generation: int) -> None:
        for entry in self.entries.values():
            self._recompute_entry_score(entry, current_generation=current_generation)

    def _recompute_entry_score(self, entry: ArchiveEntry, *, current_generation: int) -> None:
        if current_generation <= 0:
            entry.recency_score = 1.0
        else:
            entry.recency_score = max(0.0, min(1.0, float(entry.generation) / float(current_generation)))
        entry.archive_score = (
            entry.quality_score
            + self.config.robustness_weight * entry.robustness_score
            + self.config.diversity_weight * entry.diversity_score
            + self.config.recency_weight * entry.recency_score
        )

    def _prune(self) -> None:
        before = len(self.entries)
        bucket_map = self._bucket_map()
        keep_ids: set[str] = set()
        for bucket, entries in bucket_map.items():
            ordered = sorted(entries, key=lambda entry: _bucket_sort_key(entry, bucket))
            keep_ids.update(entry.candidate_id for entry in ordered[:self.config.max_per_bucket])

        if not keep_ids and self.entries:
            ordered = sorted(self.entries.values(), key=_entry_sort_key)
            keep_ids.update(entry.candidate_id for entry in ordered[:self.config.max_size])

        kept_entries = [
            entry for entry in self.entries.values()
            if entry.candidate_id in keep_ids
        ]
        kept_entries = sorted(kept_entries, key=_entry_sort_key)[:self.config.max_size]
        final_ids = {entry.candidate_id for entry in kept_entries}
        for candidate_id in list(self.entries):
            if candidate_id not in final_ids:
                self._remove_entry(candidate_id)
        self.stats["pruned_count"] += max(0, before - len(self.entries))

    def _remove_entry(self, candidate_id: str) -> None:
        self.entries.pop(candidate_id, None)

    def _bucket_map(self) -> dict[str, list[ArchiveEntry]]:
        buckets: dict[str, list[ArchiveEntry]] = {}
        for entry in self.entries.values():
            for bucket in entry.buckets:
                buckets.setdefault(bucket, []).append(entry)
        return buckets


def normalized_code_hash(code: str) -> str:
    normalized = "\n".join(line.strip() for line in str(code or "").splitlines() if line.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clear_archive_selection(candidates: list[Candidate]) -> None:
    for candidate in candidates:
        if isinstance(candidate.metrics, dict):
            candidate.metrics.pop(ARCHIVE_SELECTION_KEY, None)


def archive_selection_ids(candidates: list[Candidate]) -> set[str]:
    result: set[str] = set()
    for candidate in candidates:
        if isinstance(candidate.metrics, dict) and isinstance(candidate.metrics.get(ARCHIVE_SELECTION_KEY), dict):
            result.add(candidate.id)
    return result


def format_archive_parent_context(candidate: Candidate) -> str:
    metrics = candidate.metrics if isinstance(candidate.metrics, dict) else {}
    selection = metrics.get(ARCHIVE_SELECTION_KEY)
    if not isinstance(selection, dict):
        return ""
    lines = [
        "Archive source: yes",
        f"Archive bucket: {selection.get('primary_bucket')}",
    ]
    role = selection.get("role")
    if role:
        lines.append(f"Archive role: {role}")
    return "\n".join(lines)


def _set_archive_metadata(
        candidate: Candidate,
        *,
        in_archive: bool,
        entry: ArchiveEntry | None = None,
        reason: str | None = None,
) -> None:
    if not isinstance(candidate.metrics, dict):
        candidate.metrics = {}
    data: dict[str, Any] = {"in_archive": bool(in_archive)}
    if entry is not None:
        data.update({
            "buckets": list(entry.buckets),
            "primary_bucket": entry.primary_bucket,
            "archive_score": entry.archive_score,
            "quality_score": entry.quality_score,
            "robustness_score": entry.robustness_score,
            "diversity_score": entry.diversity_score,
        })
    if reason:
        data["rejected_reason"] = reason
    candidate.metrics[ARCHIVE_KEY] = data


def _set_archive_selection(candidate: Candidate, entry: ArchiveEntry) -> None:
    if not isinstance(candidate.metrics, dict):
        candidate.metrics = {}
    candidate.metrics[ARCHIVE_SELECTION_KEY] = {
        "primary_bucket": entry.primary_bucket,
        "buckets": list(entry.buckets),
        "archive_score": entry.archive_score,
        "bucket_score": entry.bucket_scores.get(entry.primary_bucket, entry.archive_score),
        "role": _archive_role(entry.primary_bucket),
    }


def _archive_role(bucket: str) -> str:
    if ":runtime:" in bucket:
        return "runtime specialist"
    if ":size:" in bucket or ":trucks:" in bucket or ":function:" in bucket or ":group:" in bucket:
        return "regime specialist"
    if ":mechanism:" in bucket:
        return "mechanism specialist"
    if "global" in bucket:
        return "global elite"
    return "archive specialist"


def _rank_biased_archive_choice(
        choices: list[tuple[ArchiveEntry, Candidate]],
        rng: random.Random,
) -> tuple[ArchiveEntry, Candidate]:
    ordered = sorted(choices, key=lambda item: _entry_sort_key(item[0]))
    population_size = len(ordered)
    weights_by_id = {
        entry.candidate_id: 1.0 / (rank + population_size)
        for rank, (entry, _) in enumerate(ordered, start=1)
    }
    weights = [weights_by_id[entry.candidate_id] for entry, _ in choices]
    return rng.choices(choices, weights=weights, k=1)[0]


def _entry_sort_key(entry: ArchiveEntry) -> tuple[float, float, float, int, str]:
    return (
        -entry.archive_score,
        -entry.quality_score,
        -entry.robustness_score,
        -entry.generation,
        entry.candidate_id,
    )


def _bucket_sort_key(entry: ArchiveEntry, bucket: str) -> tuple[float, float, float, int, str]:
    bucket_score = entry.bucket_scores.get(bucket, entry.archive_score)
    return (
        -bucket_score,
        -entry.archive_score,
        -entry.quality_score,
        -entry.generation,
        entry.candidate_id,
    )


def _clean_buckets(value: object) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    result: list[str] = []
    for item in values:
        bucket = " ".join(str(item).strip().split())
        if bucket and bucket not in result:
            result.append(bucket)
    return result


def _score(value: object) -> float:
    number = _finite_or_none(value)
    if number is None:
        return 0.0
    return max(0.0, min(1.0, number))


def _score_dict(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): _score(score)
        for key, item in value.items()
        if (score := _finite_or_none(item)) is not None
    }


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _status_value(candidate: Candidate) -> str:
    return str(getattr(candidate.status, "value", candidate.status)).lower()


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

import json
import math
from typing import Any

from dynagen.candidates.candidate import Candidate

VERBAL_GRADIENT_KEY = "verbal_gradient"
VERBAL_GRADIENT_VERSION = 1


def base_verbal_gradient(
        *,
        problem: str,
        candidate: Candidate,
        parents: list[Candidate],
        generation: int,
        source: str = "static",
        summary: str = "",
        preserve: list[str] | None = None,
        weaknesses: list[str] | None = None,
        next_mutations: dict[str, str] | None = None,
        avoid: list[str] | None = None,
        evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return normalize_verbal_gradient(
        {
            "version": VERBAL_GRADIENT_VERSION,
            "problem": problem,
            "source": source,
            "candidate_id": candidate.id,
            "generation": generation,
            "parent_ids": [parent.id for parent in parents],
            "score_name": candidate.score_name,
            "score_value": _finite_or_none(candidate.score_value),
            "delta_vs_best_parent": score_delta_vs_best_parent(candidate, parents),
            "summary": summary,
            "preserve": preserve or [],
            "weaknesses": weaknesses or [],
            "next_mutations": next_mutations or {},
            "avoid": avoid or [],
            "evidence": evidence or {},
        },
        fallback_problem=problem,
        fallback_candidate=candidate,
        fallback_generation=generation,
        fallback_parents=parents,
    )


def normalize_verbal_gradient(
        value: dict[str, Any],
        *,
        fallback_problem: str,
        fallback_candidate: Candidate,
        fallback_generation: int,
        fallback_parents: list[Candidate],
        source: str | None = None,
) -> dict[str, Any]:
    gradient = dict(value) if isinstance(value, dict) else {}
    next_mutations = gradient.get("next_mutations")
    if not isinstance(next_mutations, dict):
        next_mutations = {}
    normalized_next = {
        str(key): _clean_text(value)
        for key, value in next_mutations.items()
        if _clean_text(value)
    }
    return {
        "version": int(gradient.get("version") or VERBAL_GRADIENT_VERSION),
        "problem": str(gradient.get("problem") or fallback_problem),
        "source": str(source or gradient.get("source") or "static"),
        "candidate_id": str(gradient.get("candidate_id") or fallback_candidate.id),
        "generation": int(gradient.get("generation") or fallback_generation),
        "parent_ids": _clean_list(gradient.get("parent_ids")) or [parent.id for parent in fallback_parents],
        "score_name": str(gradient.get("score_name") or fallback_candidate.score_name),
        "score_value": _finite_or_none(gradient.get("score_value", fallback_candidate.score_value)),
        "delta_vs_best_parent": _finite_or_none(
            gradient.get("delta_vs_best_parent", score_delta_vs_best_parent(fallback_candidate, fallback_parents))
        ),
        "summary": _clean_text(gradient.get("summary")),
        "preserve": _clean_list(gradient.get("preserve")),
        "weaknesses": _clean_list(gradient.get("weaknesses")),
        "next_mutations": normalized_next,
        "avoid": _clean_list(gradient.get("avoid")),
        "evidence": gradient.get("evidence") if isinstance(gradient.get("evidence"), dict) else {},
    }


def parse_llm_verbal_gradient(
        text: str,
        *,
        static_gradient: dict[str, Any],
        candidate: Candidate,
        parents: list[Candidate],
        generation: int,
) -> dict[str, Any]:
    data = _json_object_from_text(text)
    merged = dict(static_gradient)
    for key in ("summary", "preserve", "weaknesses", "next_mutations", "avoid"):
        if key in data:
            merged[key] = data[key]
    evidence = dict(static_gradient.get("evidence") or {})
    llm_evidence = data.get("evidence")
    if isinstance(llm_evidence, dict):
        evidence.update(llm_evidence)
    merged["evidence"] = evidence
    return normalize_verbal_gradient(
        merged,
        fallback_problem=str(static_gradient.get("problem") or ""),
        fallback_candidate=candidate,
        fallback_generation=generation,
        fallback_parents=parents,
        source="static+llm",
    )


def candidate_has_llm_gradient(candidate: Candidate) -> bool:
    gradient = (candidate.metrics or {}).get(VERBAL_GRADIENT_KEY)
    return isinstance(gradient, dict) and "llm" in str(gradient.get("source", ""))


def get_candidate_gradient(candidate: Candidate) -> dict[str, Any] | None:
    gradient = (candidate.metrics or {}).get(VERBAL_GRADIENT_KEY)
    return gradient if isinstance(gradient, dict) else None


def set_candidate_gradient(candidate: Candidate, gradient: dict[str, Any]) -> None:
    if not isinstance(candidate.metrics, dict):
        candidate.metrics = {}
    candidate.metrics[VERBAL_GRADIENT_KEY] = gradient


def format_parent_verbal_gradients(
        parents: list[Candidate],
        *,
        strategy: str,
        max_chars: int,
) -> str:
    compact = str(strategy) == "S3" or len(parents) > 1
    blocks = []
    for parent in parents:
        block = format_candidate_verbal_gradient(parent, strategy=strategy, compact=compact)
        if block:
            blocks.append(block)
    if not blocks:
        return ""
    return _fit_parent_gradient_blocks(blocks, max_chars=max_chars)


def format_candidate_verbal_gradient(
        candidate: Candidate,
        *,
        strategy: str | None = None,
        compact: bool = False,
) -> str:
    gradient = get_candidate_gradient(candidate)
    if not gradient:
        return ""
    if compact:
        return _format_compact_candidate_verbal_gradient(candidate, gradient, strategy=strategy)
    lines = [f"Parent {candidate.id} gradient ({gradient.get('source', 'unknown')}):"]
    summary = _clean_text(gradient.get("summary"))
    if summary:
        lines.append(f"- Summary: {summary}")
    preserve = _clean_list(gradient.get("preserve"), limit=4)
    if preserve:
        lines.append(f"- Preserve: {'; '.join(preserve)}")
    weaknesses = _clean_list(gradient.get("weaknesses"), limit=4)
    if weaknesses:
        lines.append(f"- Weaknesses: {'; '.join(weaknesses)}")
    next_mutations = gradient.get("next_mutations")
    if isinstance(next_mutations, dict):
        next_step = _clean_text(next_mutations.get(str(strategy))) if strategy else ""
        if not next_step:
            next_step = _clean_text(next_mutations.get("default"))
        if next_step:
            label = f"Next {strategy} mutation" if strategy else "Next mutation"
            lines.append(f"- {label}: {next_step}")
    avoid = _clean_list(gradient.get("avoid"), limit=4)
    if avoid:
        lines.append(f"- Avoid: {'; '.join(avoid)}")
    return "\n".join(lines)


def _format_compact_candidate_verbal_gradient(
        candidate: Candidate,
        gradient: dict[str, Any],
        *,
        strategy: str | None,
) -> str:
    source = gradient.get("source", "unknown")
    summary = _field("Summary", gradient.get("summary"), max_chars=110)
    preserve = _field("Preserve", _clean_list(gradient.get("preserve"), limit=2), max_chars=100)
    weaknesses = _field("Weak", _clean_list(gradient.get("weaknesses"), limit=2), max_chars=120)
    next_mutation = ""
    next_mutations = gradient.get("next_mutations")
    if isinstance(next_mutations, dict):
        next_step = _clean_text(next_mutations.get(str(strategy))) if strategy else ""
        if not next_step:
            next_step = _clean_text(next_mutations.get("default"))
        next_mutation = _field(f"Next {strategy}" if strategy else "Next", next_step, max_chars=140)
    avoid = _field("Avoid", _clean_list(gradient.get("avoid"), limit=1), max_chars=90)
    parts = [
        f"Parent {candidate.id} gradient ({source}):",
        summary,
        preserve,
        weaknesses,
        next_mutation,
        avoid,
    ]
    return "\n".join(part for part in parts if part)


def _field(label: str, value: object, *, max_chars: int) -> str:
    if isinstance(value, (list, tuple, set)):
        text = "; ".join(_clean_list(value, limit=3))
    else:
        text = _clean_text(value)
    if not text:
        return ""
    return f"- {label}: {trim_text(text, max_chars=max_chars)}"


def _fit_parent_gradient_blocks(blocks: list[str], *, max_chars: int) -> str:
    header = "PARENT-SPECIFIC VERBAL GRADIENTS:"
    text = header + "\n\n" + "\n\n".join(blocks)
    if len(text) <= max_chars:
        return text
    if max_chars <= len(header) + 12:
        return trim_text(text, max_chars=max_chars)

    separator_chars = 2 * max(0, len(blocks) - 1)
    available = max_chars - len(header) - 2 - separator_chars
    per_block = max(120, available // max(1, len(blocks)))
    trimmed_blocks = [trim_text(block, max_chars=per_block) for block in blocks]
    text = header + "\n\n" + "\n\n".join(trimmed_blocks)
    if len(text) <= max_chars:
        return text
    return trim_text(text, max_chars=max_chars)


def build_llm_gradient_messages(
        *,
        problem: str,
        goal: str,
        focus: str,
        candidate: Candidate,
        parents: list[Candidate],
        generation: int,
        static_gradient: dict[str, Any],
) -> list[dict[str, str]]:
    evidence = {
        "generation": generation,
        "problem": problem,
        "goal": goal,
        "candidate": _candidate_snapshot(candidate),
        "parents": [_candidate_snapshot(parent) for parent in parents],
        "static_gradient": static_gradient,
    }
    user = (
        "Convert evaluator evidence into a concise parent-specific verbal gradient for future mutations. "
        "The gradient must be actionable for evolutionary code generation, not a generic review. "
        f"Optimization goal: {goal}\n"
        f"Domain focus: {focus}\n\n"
        "Return exactly one JSON object with these keys: summary, preserve, weaknesses, next_mutations, avoid. "
        "next_mutations must be an object with S1, S2, S3, and default string values. "
        "Do not include Markdown, code, or text outside JSON.\n\n"
        f"Evidence:\n{json.dumps(evidence, sort_keys=True, separators=(',', ':'))}\n"
    )
    return [
        {
            "role": "system",
            "content": "Produce compact evaluator-grounded verbal gradients for algorithm mutation.",
        },
        {"role": "user", "content": user},
    ]


def score_delta_vs_best_parent(candidate: Candidate, parents: list[Candidate]) -> float | None:
    child_score = _finite_or_none(candidate.score_value)
    parent_scores = [
        score for score in (_finite_or_none(parent.score_value) for parent in parents)
        if score is not None
    ]
    if child_score is None or not parent_scores:
        return None
    return child_score - min(parent_scores)


def metric_float(metrics: dict[str, Any], key: str) -> float | None:
    return _finite_or_none(metrics.get(key))


def best_numeric_group(groups: object, *, higher_is_better: bool) -> tuple[str, float] | None:
    if not isinstance(groups, dict):
        return None
    values = [
        (str(key), number)
        for key, value in groups.items()
        if (number := _finite_or_none(value)) is not None
    ]
    if not values:
        return None
    return max(values, key=lambda item: item[1]) if higher_is_better else min(values, key=lambda item: item[1])


def worst_numeric_group(groups: object, *, higher_is_better: bool) -> tuple[str, float] | None:
    if not isinstance(groups, dict):
        return None
    values = [
        (str(key), number)
        for key, value in groups.items()
        if (number := _finite_or_none(value)) is not None
    ]
    if not values:
        return None
    return min(values, key=lambda item: item[1]) if higher_is_better else max(values, key=lambda item: item[1])


def trim_text(text: str, *, max_chars: int) -> str:
    text = "\n".join(line.rstrip() for line in str(text).strip().splitlines())
    if len(text) <= max_chars:
        return text
    return text[:max(0, max_chars - 3)].rstrip() + "..."


def _json_object_from_text(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Verbal gradient response must be a JSON object")
    return data


def _candidate_snapshot(candidate: Candidate) -> dict[str, Any]:
    metrics = candidate.metrics or {}
    return {
        "id": candidate.id,
        "generation": candidate.generation,
        "strategy": candidate.strategy,
        "status": str(getattr(candidate.status, "value", candidate.status)),
        "score_name": candidate.score_name,
        "score_value": _finite_or_none(candidate.score_value),
        "thought": candidate.thought,
        "error_details": candidate.error_details,
        "metrics": {
            key: value for key, value in metrics.items()
            if key not in {VERBAL_GRADIENT_KEY, "records"}
        },
        "code_excerpt": candidate.code[:2500],
    }


def _clean_list(value: object, *, limit: int = 6) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    cleaned = [_clean_text(item) for item in items]
    return [item for item in cleaned if item][:limit]


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

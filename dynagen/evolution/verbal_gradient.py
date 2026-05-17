import json
import math
from typing import Any

from dynagen.candidates.candidate import Candidate

VERBAL_GRADIENT_KEY = "verbal_gradient"
VERBAL_GRADIENT_VERSION = 1


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
    return {
        "version": int(gradient.get("version") or VERBAL_GRADIENT_VERSION),
        "problem": str(gradient.get("problem") or fallback_problem),
        "source": str(source or gradient.get("source") or "llm"),
        "candidate_id": str(gradient.get("candidate_id") or fallback_candidate.id),
        "generation": int(gradient.get("generation") or fallback_generation),
        "parent_ids": _clean_list(gradient.get("parent_ids")) or [parent.id for parent in fallback_parents],
        "score_name": str(gradient.get("score_name") or fallback_candidate.score_name),
        "score_value": _finite_or_none(gradient.get("score_value", fallback_candidate.score_value)),
        "delta_vs_best_parent": _finite_or_none(
            gradient.get("delta_vs_best_parent", score_delta_vs_best_parent(fallback_candidate, fallback_parents))
        ),
        "summary": _clean_text(gradient.get("summary")),
        "aim": _clean_text(gradient.get("aim")),
        "preserve": _clean_list(gradient.get("preserve")),
        "change": _clean_list(gradient.get("change")),
        "avoid": _clean_list(gradient.get("avoid")),
        "evidence": gradient.get("evidence") if isinstance(gradient.get("evidence"), dict) else {},
    }


def parse_llm_verbal_gradient(
        text: str,
        *,
        problem: str,
        candidate: Candidate,
        parents: list[Candidate],
        generation: int,
) -> dict[str, Any]:
    data = _json_object_from_text(text)
    return normalize_verbal_gradient(
        data,
        fallback_problem=problem,
        fallback_candidate=candidate,
        fallback_generation=generation,
        fallback_parents=parents,
        source="llm",
    )


def candidate_has_llm_gradient(candidate: Candidate) -> bool:
    gradient = (candidate.metrics or {}).get(VERBAL_GRADIENT_KEY)
    return isinstance(gradient, dict) and str(gradient.get("source")) == "llm"


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
) -> str:
    blocks = []
    for parent in parents:
        block = format_candidate_verbal_gradient(parent, strategy=strategy)
        if block:
            blocks.append(block)
    if not blocks:
        return ""
    return "PARENT-SPECIFIC LLM REFLECTIONS:\n\n" + "\n\n".join(blocks)


def format_candidate_verbal_gradient(
        candidate: Candidate,
        *,
        strategy: str | None = None,
) -> str:
    gradient = get_candidate_gradient(candidate)
    if not gradient or str(gradient.get("source")) != "llm":
        return ""
    lines = [f"Parent {candidate.id} LLM reflection:"]
    summary = _clean_text(gradient.get("summary"))
    if summary:
        lines.append(f"- Summary: {summary}")
    aim = _clean_text(gradient.get("aim"))
    if aim:
        lines.append(f"- Aim: {aim}")
    preserve = _clean_list(gradient.get("preserve"))
    if preserve:
        lines.append(f"- Preserve: {'; '.join(preserve)}")
    change = _clean_list(gradient.get("change"))
    if change:
        label = f"Change for {strategy}" if strategy else "Change"
        lines.append(f"- {label}: {'; '.join(change)}")
    avoid = _clean_list(gradient.get("avoid"))
    if avoid:
        lines.append(f"- Avoid: {'; '.join(avoid)}")
    return "\n".join(lines)


def build_llm_gradient_messages(
        *,
        problem: str,
        goal: str,
        focus: str,
        candidate: Candidate,
        parents: list[Candidate],
        generation: int,
) -> list[dict[str, str]]:
    evidence = {
        "generation": generation,
        "problem": problem,
        "goal": goal,
        "candidate": _candidate_snapshot(candidate),
        "parents": [_candidate_snapshot(parent) for parent in parents],
        "delta_vs_best_parent": score_delta_vs_best_parent(candidate, parents),
    }
    user = (
        "Create one simple, aim-guided LLM reflection before this candidate is reused as a parent. "
        "Use measured evaluator evidence, parent comparison, code, thought, and errors. "
        "Do not invent unsupported weaknesses.\n"
        f"Optimization goal: {goal}\n"
        f"Domain focus: {focus}\n\n"
        "Return exactly one JSON object with these keys: summary, aim, preserve, change, avoid.\n"
        "summary: one sentence describing the measured outcome.\n"
        "aim: one concrete objective for the next mutation, directly aligned with the optimization goal.\n"
        "preserve: a short list of mechanisms worth keeping.\n"
        "change: a short list of one or two targeted changes to try next.\n"
        "avoid: a short list of failure modes or distracting edits to avoid.\n"
        "Keep every value concise. Do not include Markdown, code, or text outside JSON.\n\n"
        f"Evidence:\n{json.dumps(evidence, sort_keys=True, separators=(',', ':'))}\n"
    )
    return [
        {
            "role": "system",
            "content": "Produce concise, aim-guided LLM reflections for evolutionary code mutation.",
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
        raise ValueError("Reflection response must be a JSON object")
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
        "code": candidate.code,
    }


def _clean_list(value: object, *, limit: int | None = None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    cleaned = [_clean_text(item) for item in items]
    cleaned_items = [item for item in cleaned if item]
    if limit is None:
        return cleaned_items
    return cleaned_items[:limit]


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

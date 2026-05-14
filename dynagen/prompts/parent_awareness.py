from dynagen.candidates.candidate import Candidate
from dynagen.evolution.archive import ARCHIVE_SELECTION_KEY
from dynagen.evolution.verbal_gradient import VERBAL_GRADIENT_KEY


def render_parent_awareness(
        parents: list[Candidate],
        *,
        strategy: str,
        problem: str,
        score_label: str | None = None,
) -> str:
    if not parents:
        return ""

    label = score_label or _score_label(parents)
    best_parent = _best_scored_parent(parents)
    invalid_parents = [
        parent.id for parent in parents
        if str(parent.status) not in {"valid", "evaluated"}
    ]
    archive_parents = [
        _archive_snapshot(parent) for parent in parents
        if _archive_snapshot(parent)
    ]

    lines = [
        "PARENT AWARENESS:",
        "- Treat selected parents as evaluated evidence, not templates to copy blindly.",
        f"- Parent ranking in this prompt uses lower {label} as better unless a problem metric explicitly says otherwise.",
    ]
    if best_parent:
        lines.append(
            f"- Best available parent by current scalar score: {best_parent.id} "
            f"({label}={_format_score(best_parent.score_value)}, status={best_parent.status}, "
            f"generation={best_parent.generation}, strategy={best_parent.strategy})."
        )
    if invalid_parents:
        lines.append(
            "- Timeout, invalid, or error parents should be treated as failure evidence except for mechanisms "
            f"explicitly preserved by their verbal gradients: {', '.join(invalid_parents)}."
        )
    if archive_parents:
        lines.append(
            "- Archive-selected parents are retained specialists; preserve their bucket role only when it is compatible "
            f"with this mutation: {'; '.join(archive_parents)}."
        )

    guidance = _strategy_guidance(strategy)
    if guidance:
        lines.append(f"- Strategy-aware parent use for {strategy}: {guidance}")

    lines.append("PARENT SNAPSHOT:")
    for parent in parents:
        lines.append(f"- {_parent_snapshot(parent, label=label, best_parent=best_parent, problem=problem)}")
    return "\n".join(lines)


def _strategy_guidance(strategy: str) -> str:
    if strategy == "S1":
        return (
            "depart from the selected parent's core mechanism, but keep any proven validity, reporting, "
            "budget, or incumbent-handling behavior."
        )
    if strategy == "S2":
        return (
            "use the strongest valid parent as the backbone and make one or two measured fixes from its "
            "weaknesses and next-mutation guidance."
        )
    if strategy == "S3":
        return (
            "assign each parent a role such as backbone, complementary mechanism, specialist, or cautionary "
            "example; produce one coherent child rather than a sequential ensemble."
        )
    return ""


def _parent_snapshot(
        parent: Candidate,
        *,
        label: str,
        best_parent: Candidate | None,
        problem: str,
) -> str:
    pieces = [
        f"{parent.id}",
        f"gen={parent.generation}",
        f"strategy={parent.strategy}",
        f"status={parent.status}",
        f"{label}={_format_score(parent.score_value)}",
    ]
    relation = _score_relation(parent, best_parent)
    if relation:
        pieces.append(f"relative_score={relation}")
    archive = _archive_snapshot(parent)
    if archive:
        pieces.append(f"archive={archive}")
    gradient = _gradient_snapshot(parent)
    if gradient:
        pieces.append(f"gradient={gradient}")
    problem_metrics = _problem_metric_snapshot(parent, problem=problem)
    if problem_metrics:
        pieces.append(problem_metrics)
    return "; ".join(pieces)


def _gradient_snapshot(parent: Candidate) -> str:
    metrics = parent.metrics if isinstance(parent.metrics, dict) else {}
    gradient = metrics.get(VERBAL_GRADIENT_KEY)
    if not isinstance(gradient, dict):
        return "none"
    source = str(gradient.get("source") or "unknown")
    summary = str(gradient.get("summary") or "").strip()
    if summary:
        return f"{source}; summary={summary}"
    return source


def _problem_metric_snapshot(parent: Candidate, *, problem: str) -> str:
    metrics = parent.metrics if isinstance(parent.metrics, dict) else {}
    if problem == "tsp":
        fields = ("timeout_fraction", "mean_gap", "worst_gap", "mean_tour_length")
    elif problem == "bbob":
        fields = ("timeout_fraction", "mean_aocc", "mean_final_error", "worst_final_error")
    elif problem == "dvrp":
        fields = ("timeout_fraction", "mean_gap", "worst_gap", "mean_makespan", "mean_waits")
    else:
        fields = ("timeout_fraction",)
    values = [
        f"{field}={metrics[field]}"
        for field in fields
        if field in metrics and metrics[field] is not None
    ]
    return "metrics=" + ", ".join(values) if values else ""


def _archive_snapshot(parent: Candidate) -> str:
    metrics = parent.metrics if isinstance(parent.metrics, dict) else {}
    selection = metrics.get(ARCHIVE_SELECTION_KEY)
    if not isinstance(selection, dict):
        return ""
    parts = []
    role = selection.get("role")
    bucket = selection.get("primary_bucket")
    if role:
        parts.append(str(role))
    if bucket:
        parts.append(str(bucket))
    archive_score = selection.get("archive_score")
    if archive_score is not None:
        parts.append(f"score={_format_score(archive_score)}")
    return "/".join(parts)


def _score_label(parents: list[Candidate]) -> str:
    for parent in parents:
        if parent.score_name:
            return parent.score_name
    return "score"


def _best_scored_parent(parents: list[Candidate]) -> Candidate | None:
    scored = [
        parent for parent in parents
        if _finite_float(parent.score_value) is not None
    ]
    if not scored:
        return None
    return min(scored, key=lambda parent: (_finite_float(parent.score_value), parent.id))


def _score_relation(parent: Candidate, best_parent: Candidate | None) -> str:
    score = _finite_float(parent.score_value)
    best_score = _finite_float(best_parent.score_value if best_parent else None)
    if score is None or best_score is None:
        return ""
    if parent.id == best_parent.id:
        return "best"
    delta = score - best_score
    if abs(delta) <= max(1e-9, abs(best_score) * 0.005):
        return f"near_best_delta={_format_score(delta)}"
    return f"delta={_format_score(delta)}"


def _format_score(value: object) -> str:
    number = _finite_float(value)
    return "unknown" if number is None else f"{number:.6g}"


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number

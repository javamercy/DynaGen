"""
BBOB_sensitivity_analysis.py
============================

Ablation / sensitivity analysis across four DynaGen BBOB runs that differ only
in which auxiliary features are enabled:

    * base             - history off, verbal_gradients off
    * only_memory      - history on,  verbal_gradients off
    * only_reflection  - history off, verbal_gradients on
    * full             - history on,  verbal_gradients on

This is a 2x2 factorial over two switches:

    M = memory   (config.evolution.history.enabled)
    R = reflection (config.evolution.verbal_gradients.enabled)

For every metric of interest (test mean AOCC, per-group AOCC, ...) the script
reports:

    main_effect(M)    = mean(M=on)  - mean(M=off)
    main_effect(R)    = mean(R=on)  - mean(R=off)
    interaction(MxR)  = (full - only_reflection) - (only_memory - base)
                      = how much memory's effect changes when reflection is on
                        (>0 superadditive, <0 subadditive)

Usage:
    python BBOB_sensitivity_analysis.py
    python BBOB_sensitivity_analysis.py \
        --base       runs/bbob/<base_run> \
        --memory     runs/bbob/<only_memory_run> \
        --reflection runs/bbob/<only_reflection_run> \
        --full       runs/bbob/<full_run> \
        --out        bbob_sensitivity_out

Outputs (written to ``--out`` directory):
    summary.json             machine-readable summary
    summary.md               human-readable report
    per_function_aocc.csv    24 x 4 table of test AOCC per function
    per_function_aocc.png    grouped bar chart, one bar per run per function
    per_group_aocc.csv       per-BBOB-group test AOCC table
    per_group_aocc.png       per-group bar chart
    convergence.csv          best-so-far train AOCC per eval index, 4 columns
    convergence.png          overlay of all four convergence curves
    effects.csv              main effects + interaction per metric
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Defaults (the four runs the user is currently analysing)
# ---------------------------------------------------------------------------
DEFAULT_RUNS: dict[str, str] = {
    "base": "runs/bbob/20260525_145727_bbob_pop_5_gen_20_base",
    "only_memory": "runs/bbob/20260525_210453_bbob_pop_5_gen_20_only_memory",
    "only_reflection": "runs/bbob/20260525_162140_bbob_pop_5_gen_20_only_reflection",
    "full": "runs/bbob/20260525_000245_bbob_pop_5_gen_20_verbal_3",
}

# Cell flags for the 2x2: (memory, reflection)
RUN_CELLS: dict[str, tuple[bool, bool]] = {
    "base": (False, False),
    "only_memory": (True, False),
    "only_reflection": (False, True),
    "full": (True, True),
}

# Pretty labels for plots / report.
RUN_LABELS: dict[str, str] = {
    "base": "base (M=off, R=off)",
    "only_memory": "only_memory (M=on, R=off)",
    "only_reflection": "only_reflection (M=off, R=on)",
    "full": "full (M=on, R=on)",
}

RUN_COLORS: dict[str, str] = {
    "base": "tab:gray",
    "only_memory": "tab:blue",
    "only_reflection": "tab:orange",
    "full": "tab:green",
}

BBOB_FUNCTIONS: list[int] = list(range(1, 25))

BBOB_GROUPS: list[str] = [
    "separable",
    "low_moderate_conditioning",
    "high_conditioning_unimodal",
    "multimodal_strong_global_structure",
    "multimodal_weak_global_structure",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_run(run_dir: Path) -> dict[str, Any]:
    """Read a DynaGen BBOB run directory and extract the values used below."""
    config = json.loads((run_dir / "config.json").read_text())
    test_result = json.loads((run_dir / "test_result.json").read_text())
    llm_calls = {}
    llm_calls_path = run_dir / "llm_calls.json"
    if llm_calls_path.exists():
        llm_calls = json.loads(llm_calls_path.read_text())

    # Per-candidate stream, chronological.
    candidates: list[dict[str, Any]] = []
    for path in sorted((run_dir / "candidates").glob("cand_*.json")):
        candidates.append(json.loads(path.read_text()))
    candidates.sort(key=lambda c: c.get("created_at", ""))

    train_curve: list[float] = []
    valid_train: list[float] = []
    best_so_far = -math.inf
    best_train_id: str | None = None
    for cand in candidates:
        score = cand.get("mean_aocc")
        if score is None or cand.get("status") != "valid":
            train_curve.append(best_so_far if best_so_far > -math.inf else float("nan"))
            continue
        valid_train.append(score)
        if score > best_so_far:
            best_so_far = score
            best_train_id = cand.get("id")
        train_curve.append(best_so_far)

    test_metrics = test_result.get("metrics", {}) or {}
    per_function_test = {
        int(k): float(v)
        for k, v in (test_metrics.get("aocc_by_function") or {}).items()
    }
    per_group_test = {
        str(k): float(v) for k, v in (test_metrics.get("aocc_by_group") or {}).items()
    }

    history_cfg = (config.get("evolution") or {}).get("history") or {}
    verbal_cfg = (config.get("evolution") or {}).get("verbal_gradients") or {}

    return {
        "name": config.get("name", run_dir.name),
        "run_dir": str(run_dir),
        "config_memory_enabled": bool(history_cfg.get("enabled", False)),
        "config_reflection_enabled": bool(verbal_cfg.get("enabled", False)),
        "reflection_cadence": verbal_cfg.get("llm_every_n_generations"),
        "candidates_total": len(candidates),
        "candidates_valid": len(valid_train),
        "best_train_aocc": max(valid_train) if valid_train else float("nan"),
        "best_train_id": best_train_id,
        "final_selected_id": test_result.get("candidate_id"),
        "final_test_mean_aocc": test_result.get("mean_aocc"),
        "final_test_best_aocc": test_metrics.get("best_aocc"),
        "final_test_median_aocc": test_metrics.get("median_aocc"),
        "final_test_mean_final_error": test_metrics.get("mean_final_error"),
        "per_function_test": per_function_test,
        "per_group_test": per_group_test,
        "train_best_so_far_curve": train_curve,
        "llm_total_api_calls": llm_calls.get("total_api_calls"),
        "llm_candidate_calls": llm_calls.get("candidate_generation_calls"),
        "llm_feedback_calls": llm_calls.get("feedback_calls"),
        "llm_reflection_calls": (llm_calls.get("reflection") or {}).get(
            "calls"
        ) if isinstance(llm_calls.get("reflection"), dict) else llm_calls.get(
            "reflection_calls"
        ),
    }


def verify_cell(label: str, run: dict[str, Any]) -> list[str]:
    """Cross-check that the run's config matches the assumed (M, R) cell."""
    expected_m, expected_r = RUN_CELLS[label]
    warnings: list[str] = []
    if run["config_memory_enabled"] != expected_m:
        warnings.append(
            f"[{label}] config.history.enabled={run['config_memory_enabled']} but expected {expected_m}"
        )
    if run["config_reflection_enabled"] != expected_r:
        warnings.append(
            f"[{label}] config.verbal_gradients.enabled={run['config_reflection_enabled']} but expected {expected_r}"
        )
    return warnings


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------
def effect_decomposition(
    cells: dict[str, float],
) -> dict[str, float]:
    """Given a metric value per cell, return main effects + interaction.

    cells must contain keys: base, only_memory, only_reflection, full.
    """
    base = cells["base"]
    mem = cells["only_memory"]
    ref = cells["only_reflection"]
    full = cells["full"]
    # Means over the "memory on" rows vs "memory off" rows (averaged across R).
    mean_m_on = 0.5 * (mem + full)
    mean_m_off = 0.5 * (base + ref)
    mean_r_on = 0.5 * (ref + full)
    mean_r_off = 0.5 * (base + mem)
    main_m = mean_m_on - mean_m_off
    main_r = mean_r_on - mean_r_off
    # Interaction: how much memory's effect depends on reflection state.
    interaction = (full - ref) - (mem - base)
    return {
        "base": base,
        "only_memory": mem,
        "only_reflection": ref,
        "full": full,
        "delta_memory_vs_base": mem - base,
        "delta_reflection_vs_base": ref - base,
        "delta_full_vs_base": full - base,
        "main_effect_memory": main_m,
        "main_effect_reflection": main_r,
        "interaction_MxR": interaction,
    }


def collect_metric(
    runs: dict[str, dict[str, Any]], key: str
) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, run in runs.items():
        v = run.get(key)
        out[label] = float("nan") if v is None else float(v)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_convergence(runs: dict[str, dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in DEFAULT_RUNS:
        curve = runs[label]["train_best_so_far_curve"]
        ax.plot(
            range(1, len(curve) + 1),
            curve,
            label=RUN_LABELS[label],
            color=RUN_COLORS[label],
            linewidth=1.6,
        )
    ax.set_xlabel("Candidate evaluation index (chronological)")
    ax.set_ylabel("Best-so-far mean AOCC (train pool)")
    ax.set_title("BBOB convergence by ablation cell")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_per_function(runs: dict[str, dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.2
    x = np.arange(len(BBOB_FUNCTIONS))
    labels = list(DEFAULT_RUNS.keys())
    for i, label in enumerate(labels):
        per_fn = runs[label]["per_function_test"]
        vals = [per_fn.get(fn, np.nan) for fn in BBOB_FUNCTIONS]
        offset = (i - (len(labels) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=RUN_LABELS[label],
            color=RUN_COLORS[label],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"f{fn}" for fn in BBOB_FUNCTIONS], rotation=45, ha="right")
    ax.set_ylabel("Mean AOCC on test instances")
    ax.set_title("Per-function test AOCC across ablation cells")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_per_group(runs: dict[str, dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    x = np.arange(len(BBOB_GROUPS))
    labels = list(DEFAULT_RUNS.keys())
    for i, label in enumerate(labels):
        per_group = runs[label]["per_group_test"]
        vals = [per_group.get(g, np.nan) for g in BBOB_GROUPS]
        offset = (i - (len(labels) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=RUN_LABELS[label],
            color=RUN_COLORS[label],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace("_", "\n") for g in BBOB_GROUPS], fontsize=9
    )
    ax.set_ylabel("Mean AOCC on test instances")
    ax.set_title("Per-group test AOCC across ablation cells")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def write_convergence_csv(
    runs: dict[str, dict[str, Any]], out_path: Path
) -> None:
    labels = list(DEFAULT_RUNS.keys())
    curves = {label: runs[label]["train_best_so_far_curve"] for label in labels}
    n = max(len(c) for c in curves.values())
    header = ["eval_index"] + [f"{label}_best_so_far_aocc" for label in labels]
    lines = [",".join(header)]
    for i in range(n):
        row = [str(i + 1)]
        for label in labels:
            c = curves[label]
            v = c[i] if i < len(c) else None
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                row.append("")
            else:
                row.append(f"{float(v):.6f}")
        lines.append(",".join(row))
    out_path.write_text("\n".join(lines) + "\n")


def write_per_function_csv(
    runs: dict[str, dict[str, Any]], out_path: Path
) -> None:
    labels = list(DEFAULT_RUNS.keys())
    header = ["function"] + labels + [
        "delta_memory_vs_base",
        "delta_reflection_vs_base",
        "delta_full_vs_base",
        "main_effect_memory",
        "main_effect_reflection",
        "interaction_MxR",
    ]
    lines = [",".join(header)]
    for fn in BBOB_FUNCTIONS:
        vals = {label: runs[label]["per_function_test"].get(fn) for label in labels}
        if any(v is None for v in vals.values()):
            row = [f"f{fn}"] + [
                "" if vals[label] is None else f"{vals[label]:.6f}" for label in labels
            ] + [""] * 6
            lines.append(",".join(row))
            continue
        eff = effect_decomposition({k: float(v) for k, v in vals.items()})
        row = [f"f{fn}"] + [f"{vals[label]:.6f}" for label in labels] + [
            f"{eff['delta_memory_vs_base']:.6f}",
            f"{eff['delta_reflection_vs_base']:.6f}",
            f"{eff['delta_full_vs_base']:.6f}",
            f"{eff['main_effect_memory']:.6f}",
            f"{eff['main_effect_reflection']:.6f}",
            f"{eff['interaction_MxR']:.6f}",
        ]
        lines.append(",".join(row))
    out_path.write_text("\n".join(lines) + "\n")


def write_per_group_csv(
    runs: dict[str, dict[str, Any]], out_path: Path
) -> None:
    labels = list(DEFAULT_RUNS.keys())
    header = ["group"] + labels + [
        "delta_memory_vs_base",
        "delta_reflection_vs_base",
        "delta_full_vs_base",
        "main_effect_memory",
        "main_effect_reflection",
        "interaction_MxR",
    ]
    lines = [",".join(header)]
    for g in BBOB_GROUPS:
        vals = {label: runs[label]["per_group_test"].get(g) for label in labels}
        if any(v is None for v in vals.values()):
            row = [g] + [
                "" if vals[label] is None else f"{vals[label]:.6f}" for label in labels
            ] + [""] * 6
            lines.append(",".join(row))
            continue
        eff = effect_decomposition({k: float(v) for k, v in vals.items()})
        row = [g] + [f"{vals[label]:.6f}" for label in labels] + [
            f"{eff['delta_memory_vs_base']:.6f}",
            f"{eff['delta_reflection_vs_base']:.6f}",
            f"{eff['delta_full_vs_base']:.6f}",
            f"{eff['main_effect_memory']:.6f}",
            f"{eff['main_effect_reflection']:.6f}",
            f"{eff['interaction_MxR']:.6f}",
        ]
        lines.append(",".join(row))
    out_path.write_text("\n".join(lines) + "\n")


def write_effects_csv(effects: dict[str, dict[str, float]], out_path: Path) -> None:
    cols = [
        "metric",
        "base",
        "only_memory",
        "only_reflection",
        "full",
        "delta_memory_vs_base",
        "delta_reflection_vs_base",
        "delta_full_vs_base",
        "main_effect_memory",
        "main_effect_reflection",
        "interaction_MxR",
    ]
    lines = [",".join(cols)]
    for metric, eff in effects.items():
        row = [metric]
        for c in cols[1:]:
            v = eff.get(c)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                row.append("")
            else:
                row.append(f"{v:.6f}")
        lines.append(",".join(row))
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x):
            return "n/a"
        return f"{x:.{digits}f}"
    return str(x)


def write_markdown(
    runs: dict[str, dict[str, Any]],
    effects: dict[str, dict[str, float]],
    warnings: list[str],
    out_path: Path,
) -> None:
    labels = list(DEFAULT_RUNS.keys())
    lines: list[str] = []
    lines.append("# BBOB sensitivity analysis (2x2: memory x reflection)\n")

    lines.append("## Runs\n")
    lines.append("| Cell | Memory | Reflection | Cadence | Run dir |")
    lines.append("|---|:-:|:-:|:-:|---|")
    for label in labels:
        run = runs[label]
        m = "on" if run["config_memory_enabled"] else "off"
        r = "on" if run["config_reflection_enabled"] else "off"
        cad = run["reflection_cadence"] if run["config_reflection_enabled"] else "-"
        lines.append(
            f"| {label} | {m} | {r} | {cad} | `{run['run_dir']}` |"
        )
    lines.append("")

    if warnings:
        lines.append("## Config-vs-cell consistency warnings\n")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Headline metrics\n")
    lines.append(
        "| Metric | base | only_memory | only_reflection | full | "
        "Δ(memory) | Δ(reflection) | Main(M) | Main(R) | Interaction(MxR) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    headline_metrics = [
        ("Test mean AOCC", "final_test_mean_aocc"),
        ("Test median AOCC", "final_test_median_aocc"),
        ("Test best AOCC", "final_test_best_aocc"),
        ("Test mean final error", "final_test_mean_final_error"),
        ("Best train AOCC (any cand)", "best_train_aocc"),
        ("LLM total API calls", "llm_total_api_calls"),
    ]
    for pretty, key in headline_metrics:
        eff = effects.get(key)
        if eff is None:
            continue
        lines.append(
            "| {pretty} | {b} | {m} | {r} | {f} | {dm} | {dr} | {mm} | {mr} | {inter} |".format(
                pretty=pretty,
                b=fmt(eff["base"]),
                m=fmt(eff["only_memory"]),
                r=fmt(eff["only_reflection"]),
                f=fmt(eff["full"]),
                dm=fmt(eff["delta_memory_vs_base"]),
                dr=fmt(eff["delta_reflection_vs_base"]),
                mm=fmt(eff["main_effect_memory"]),
                mr=fmt(eff["main_effect_reflection"]),
                inter=fmt(eff["interaction_MxR"]),
            )
        )
    lines.append("")

    # Cost vs gain.
    base = runs["base"]
    full = runs["full"]
    base_calls = runs["base"].get("llm_total_api_calls") or 0
    lines.append("## LLM cost vs AOCC gain\n")
    lines.append(
        "| Cell | API calls | Extra calls vs base | Test AOCC | Δ vs base | "
        "AOCC gain per extra call |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label in labels:
        run = runs[label]
        calls = run.get("llm_total_api_calls") or 0
        extra = calls - base_calls
        aocc = run.get("final_test_mean_aocc")
        delta = (aocc - base["final_test_mean_aocc"]) if (
            aocc is not None and base["final_test_mean_aocc"] is not None
        ) else None
        per_call = (delta / extra) if (delta is not None and extra > 0) else None
        lines.append(
            "| {l} | {c} | {e} | {a} | {d} | {p} |".format(
                l=label,
                c=calls if calls else "n/a",
                e=extra if calls else "n/a",
                a=fmt(aocc),
                d=fmt(delta),
                p="-" if per_call is None else f"{per_call:.6f}",
            )
        )
    lines.append("")

    lines.append("## Per-BBOB-group test AOCC\n")
    lines.append(
        "| Group | base | only_memory | only_reflection | full | "
        "Main(M) | Main(R) | Interaction(MxR) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for g in BBOB_GROUPS:
        vals = {label: runs[label]["per_group_test"].get(g) for label in labels}
        if any(v is None for v in vals.values()):
            continue
        eff = effect_decomposition({k: float(v) for k, v in vals.items()})
        lines.append(
            "| {g} | {b} | {m} | {r} | {f} | {mm} | {mr} | {inter} |".format(
                g=g,
                b=fmt(eff["base"]),
                m=fmt(eff["only_memory"]),
                r=fmt(eff["only_reflection"]),
                f=fmt(eff["full"]),
                mm=fmt(eff["main_effect_memory"]),
                mr=fmt(eff["main_effect_reflection"]),
                inter=fmt(eff["interaction_MxR"]),
            )
        )
    lines.append("")

    lines.append("## Per-function test AOCC\n")
    lines.append(
        "| f | base | only_memory | only_reflection | full | "
        "Main(M) | Main(R) | Interaction(MxR) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fn in BBOB_FUNCTIONS:
        vals = {label: runs[label]["per_function_test"].get(fn) for label in labels}
        if any(v is None for v in vals.values()):
            continue
        eff = effect_decomposition({k: float(v) for k, v in vals.items()})
        lines.append(
            "| f{fn} | {b} | {m} | {r} | {f} | {mm} | {mr} | {inter} |".format(
                fn=fn,
                b=fmt(eff["base"]),
                m=fmt(eff["only_memory"]),
                r=fmt(eff["only_reflection"]),
                f=fmt(eff["full"]),
                mm=fmt(eff["main_effect_memory"]),
                mr=fmt(eff["main_effect_reflection"]),
                inter=fmt(eff["interaction_MxR"]),
            )
        )
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- AOCC = Area Over the Convergence Curve, normalised to [0, 1] using "
        "lower bound 1e-8 and upper bound 100 (IOH default). Higher is better."
    )
    lines.append(
        "- Main effects average across the other switch: e.g. Main(M) = "
        "0.5*((only_memory - base) + (full - only_reflection))."
    )
    lines.append(
        "- Interaction(MxR) = (full - only_reflection) - (only_memory - base). "
        ">0 means the two features reinforce each other (superadditive); "
        "<0 means they partially substitute (subadditive)."
    )
    if runs["full"]["reflection_cadence"] != runs["only_reflection"]["reflection_cadence"]:
        lines.append(
            "- **Caveat**: `full` reflects every "
            f"{runs['full']['reflection_cadence']} generations while "
            f"`only_reflection` reflects every "
            f"{runs['only_reflection']['reflection_cadence']} generations. "
            "Reflection cadence is therefore confounded with memory in this design."
        )
    lines.append(
        "- Each cell uses a single seed; effect estimates carry no uncertainty bars. "
        "Treat magnitudes as directional rather than statistically significant."
    )

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_RUNS["base"])
    parser.add_argument("--memory", default=DEFAULT_RUNS["only_memory"])
    parser.add_argument("--reflection", default=DEFAULT_RUNS["only_reflection"])
    parser.add_argument("--full", default=DEFAULT_RUNS["full"])
    parser.add_argument("--out", default="bbob_sensitivity_out")
    args = parser.parse_args()

    paths = {
        "base": Path(args.base).expanduser().resolve(),
        "only_memory": Path(args.memory).expanduser().resolve(),
        "only_reflection": Path(args.reflection).expanduser().resolve(),
        "full": Path(args.full).expanduser().resolve(),
    }
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for label, p in paths.items():
        runs[label] = load_run(p)
        warnings.extend(verify_cell(label, runs[label]))

    # Build effect decompositions for the scalar metrics we care about.
    scalar_metrics = [
        "final_test_mean_aocc",
        "final_test_median_aocc",
        "final_test_best_aocc",
        "final_test_mean_final_error",
        "best_train_aocc",
        "llm_total_api_calls",
    ]
    effects: dict[str, dict[str, float]] = {}
    for key in scalar_metrics:
        cells = collect_metric(runs, key)
        if any(math.isnan(v) for v in cells.values()):
            effects[key] = {"base": cells["base"], "only_memory": cells["only_memory"],
                            "only_reflection": cells["only_reflection"], "full": cells["full"],
                            "delta_memory_vs_base": float("nan"),
                            "delta_reflection_vs_base": float("nan"),
                            "delta_full_vs_base": float("nan"),
                            "main_effect_memory": float("nan"),
                            "main_effect_reflection": float("nan"),
                            "interaction_MxR": float("nan")}
        else:
            effects[key] = effect_decomposition(cells)

    # Per-group / per-function effect tables are written via dedicated CSVs.
    summary = {
        "runs": {
            label: {
                k: v
                for k, v in run.items()
                if k != "train_best_so_far_curve"
            }
            for label, run in runs.items()
        },
        "effects": effects,
        "warnings": warnings,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_markdown(runs, effects, warnings, out_dir / "summary.md")
    write_per_function_csv(runs, out_dir / "per_function_aocc.csv")
    write_per_group_csv(runs, out_dir / "per_group_aocc.csv")
    write_convergence_csv(runs, out_dir / "convergence.csv")
    write_effects_csv(effects, out_dir / "effects.csv")
    plot_convergence(runs, out_dir / "convergence.png")
    plot_per_function(runs, out_dir / "per_function_aocc.png")
    plot_per_group(runs, out_dir / "per_group_aocc.png")

    # Console summary.
    print("BBOB sensitivity analysis (memory x reflection)")
    print("-" * 64)
    for label in DEFAULT_RUNS:
        run = runs[label]
        print(
            f"  {label:<16s}  test AOCC = {run['final_test_mean_aocc']:.4f}  "
            f"calls = {run.get('llm_total_api_calls') or 'n/a'}"
        )
    aocc_eff = effects["final_test_mean_aocc"]
    print()
    print(f"  Δ(memory)      = {aocc_eff['delta_memory_vs_base']:+.4f}")
    print(f"  Δ(reflection)  = {aocc_eff['delta_reflection_vs_base']:+.4f}")
    print(f"  Δ(full)        = {aocc_eff['delta_full_vs_base']:+.4f}")
    print(f"  Main(M)        = {aocc_eff['main_effect_memory']:+.4f}")
    print(f"  Main(R)        = {aocc_eff['main_effect_reflection']:+.4f}")
    print(f"  Interaction    = {aocc_eff['interaction_MxR']:+.4f}")
    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    print(f"\nWrote results to: {out_dir}")


if __name__ == "__main__":
    main()

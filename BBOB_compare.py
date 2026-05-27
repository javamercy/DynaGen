"""
BBOB_compare.py
================

Compare a DynaGen BBOB run against a LLaMEA baseline run on the same problem
suite (24 BBOB functions, dim=5, budget=10k).

Two metrics are reported, matching the LLaMEA paper (van Stein & Back, 2024):

    * AOCC  - Area Over the Convergence Curve, normalised to [0, 1].
              Higher is better.  Each entry in DynaGen / LLaMEA uses the
              same IOH-defined AOCC (lower bound 1e-8, upper bound 100).
    * best fitness - the maximum AOCC achieved by any candidate produced
              during the search (train pool), plus the AOCC of the
              finally-selected candidate evaluated on the held-out
              test pool.

By default the two directories hard-coded below are compared, but both can
be overridden on the command line.

Usage:
    python BBOB_compare.py
    python BBOB_compare.py --dynagen <run_dir> --llamea <exp_dir> \
                           --out compare_out

Outputs (written to ``--out`` directory, default ``bbob_compare_out``):
    summary.json             machine-readable summary
    summary.md               human-readable summary
    convergence.png          best-so-far AOCC vs evaluation index
    convergence.csv          best-so-far AOCC per eval index (both runs)
    per_function_aocc.png    per-function final AOCC bar chart
    per_function_aocc.csv    per-function final AOCC table
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Defaults (the two directories the user asked us to compare)
# ---------------------------------------------------------------------------
DEFAULT_DYNAGEN = (
    "runs/bbob/20260525_000245_bbob_pop_5_gen_20_verbal_3"
)
DEFAULT_LLAMEA = (
    "baselines/LLaMEA/exp-05-24_204505-LLaMEA-deepseek-v4-flash-bbob-100_gens"
)
BBOB_FUNCTIONS = list(range(1, 25))


# ---------------------------------------------------------------------------
# DynaGen loading
# ---------------------------------------------------------------------------
def load_dynagen(run_dir: Path) -> dict[str, Any]:
    """Read a DynaGen BBOB run directory."""
    config = json.loads((run_dir / "config.json").read_text())
    test_result = json.loads((run_dir / "test_result.json").read_text())

    # Per-candidate stream (chronological, indexed by creation order).
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

    # Per-generation best train AOCC.
    gen_dir = run_dir / "generations"
    per_generation: list[dict[str, Any]] = []
    if gen_dir.is_dir():
        for path in sorted(gen_dir.glob("generation_*/summary.json")):
            data = json.loads(path.read_text())
            per_generation.append(
                {
                    "generation": data.get("generation"),
                    "best_candidate_id": data.get("best_candidate_id"),
                    "best_mean_aocc": data.get("best_mean_aocc"),
                }
            )

    # Final selected candidate's per-function AOCC (test pool).
    per_function_test = {
        int(k): float(v)
        for k, v in (test_result.get("metrics", {}).get("aocc_by_function") or {}).items()
    }

    return {
        "name": config.get("name", run_dir.name),
        "run_dir": str(run_dir),
        "config": config,
        "test_result": test_result,
        "candidates_total": len(candidates),
        "candidates_valid": len(valid_train),
        "best_train_aocc": max(valid_train) if valid_train else float("nan"),
        "best_train_id": best_train_id,
        "final_selected_id": test_result.get("candidate_id"),
        "final_test_mean_aocc": test_result.get("mean_aocc"),
        "final_test_best_aocc": test_result.get("metrics", {}).get("best_aocc"),
        "final_test_mean_final_error": test_result.get("metrics", {}).get(
            "mean_final_error"
        ),
        "final_test_median_aocc": test_result.get("metrics", {}).get("median_aocc"),
        "per_function_test": per_function_test,
        "train_best_so_far_curve": train_curve,  # length == #candidates
        "per_generation": per_generation,
    }


# ---------------------------------------------------------------------------
# LLaMEA loading
# ---------------------------------------------------------------------------
def load_llamea(exp_dir: Path) -> dict[str, Any]:
    """Read a LLaMEA experiment directory."""
    log_path = exp_dir / "log.jsonl"
    sol_path = exp_dir / "solutions.jsonl"

    entries: list[dict[str, Any]] = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    train_curve: list[float] = []
    valid_train: list[float] = []
    best_so_far = -math.inf
    best_train_id: str | None = None
    best_test_aucs: list[float] | None = None
    for entry in entries:
        fitness = entry.get("fitness")
        # LLaMEA stores failed evals as -inf.
        if fitness is None or not isinstance(fitness, (int, float)) or fitness == float(
            "-inf"
        ) or math.isnan(fitness):
            train_curve.append(best_so_far if best_so_far > -math.inf else float("nan"))
            continue
        valid_train.append(fitness)
        if fitness > best_so_far:
            best_so_far = fitness
            best_train_id = entry.get("id")
            meta = entry.get("metadata") or {}
            best_test_aucs = list(meta.get("test_aucs") or []) or None
        train_curve.append(best_so_far)

    # Per-function AOCC for the best-on-train candidate (held-out test pool).
    # LLaMEA stores `test_aucs` as a flat list of length 24 * n_test_instances
    # * n_reps (e.g. 24 * 2 * 3 = 144).  We average per function id.
    per_function_test: dict[int, float] = {}
    if best_test_aucs:
        n_total = len(best_test_aucs)
        if n_total % len(BBOB_FUNCTIONS) == 0:
            per_fn = n_total // len(BBOB_FUNCTIONS)
            arr = np.asarray(best_test_aucs, dtype=float).reshape(
                len(BBOB_FUNCTIONS), per_fn
            )
            per_function_test = {
                fn: float(arr[i].mean()) for i, fn in enumerate(BBOB_FUNCTIONS)
            }

    final_solution: dict[str, Any] | None = None
    if sol_path.exists():
        with sol_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    final_solution = json.loads(line)

    final_test_mean = None
    final_train_mean = None
    final_id = None
    if final_solution is not None:
        final_test_mean = final_solution.get("test_fitness")
        final_train_mean = final_solution.get("train_fitness") or final_solution.get(
            "fitness"
        )
        final_id = final_solution.get("id")

    return {
        "name": exp_dir.name,
        "exp_dir": str(exp_dir),
        "candidates_total": len(entries),
        "candidates_valid": len(valid_train),
        "best_train_aocc": max(valid_train) if valid_train else float("nan"),
        "best_train_id": best_train_id,
        "final_selected_id": final_id,
        "final_train_mean_aocc": final_train_mean,
        "final_test_mean_aocc": final_test_mean,
        "per_function_test": per_function_test,
        "train_best_so_far_curve": train_curve,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_convergence(
    dynagen: dict[str, Any], llamea: dict[str, Any], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    dg_curve = dynagen["train_best_so_far_curve"]
    la_curve = llamea["train_best_so_far_curve"]
    ax.plot(
        range(1, len(dg_curve) + 1),
        dg_curve,
        label=f"DynaGen ({dynagen['name']})",
        color="tab:blue",
    )
    ax.plot(
        range(1, len(la_curve) + 1),
        la_curve,
        label=f"LLaMEA ({llamea['name']})",
        color="tab:orange",
    )
    ax.set_xlabel("Candidate evaluation index (chronological)")
    ax.set_ylabel("Best-so-far mean AOCC (train pool)")
    ax.set_title("BBOB convergence: DynaGen vs LLaMEA")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_per_function(
    dynagen: dict[str, Any], llamea: dict[str, Any], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.4
    x = np.arange(len(BBOB_FUNCTIONS))
    dg_vals = [dynagen["per_function_test"].get(fn, np.nan) for fn in BBOB_FUNCTIONS]
    la_vals = [llamea["per_function_test"].get(fn, np.nan) for fn in BBOB_FUNCTIONS]
    ax.bar(x - width / 2, dg_vals, width, label="DynaGen (selected)", color="tab:blue")
    ax.bar(x + width / 2, la_vals, width, label="LLaMEA (best)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"f{fn}" for fn in BBOB_FUNCTIONS], rotation=45, ha="right")
    ax.set_ylabel("Mean AOCC on test instances")
    ax.set_title("Per-function AOCC: DynaGen vs LLaMEA")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_summary(
    dynagen: dict[str, Any], llamea: dict[str, Any]
) -> dict[str, Any]:
    dg_test = dynagen["per_function_test"]
    la_test = llamea["per_function_test"]
    per_function: list[dict[str, Any]] = []
    dg_wins = la_wins = ties = 0
    for fn in BBOB_FUNCTIONS:
        dg = dg_test.get(fn)
        la = la_test.get(fn)
        winner = None
        if dg is not None and la is not None:
            if abs(dg - la) < 1e-9:
                winner = "tie"
                ties += 1
            elif dg > la:
                winner = "dynagen"
                dg_wins += 1
            else:
                winner = "llamea"
                la_wins += 1
        per_function.append(
            {
                "function": fn,
                "dynagen_aocc": dg,
                "llamea_aocc": la,
                "delta_dynagen_minus_llamea": (
                    (dg - la) if (dg is not None and la is not None) else None
                ),
                "winner": winner,
            }
        )

    dg_mean = float(np.nanmean([v for v in dg_test.values()])) if dg_test else None
    la_mean = float(np.nanmean([v for v in la_test.values()])) if la_test else None

    return {
        "dynagen": {
            "name": dynagen["name"],
            "run_dir": dynagen["run_dir"],
            "candidates_total": dynagen["candidates_total"],
            "candidates_valid": dynagen["candidates_valid"],
            "best_train_aocc": dynagen["best_train_aocc"],
            "best_train_id": dynagen["best_train_id"],
            "final_selected_id": dynagen["final_selected_id"],
            "final_test_mean_aocc": dynagen["final_test_mean_aocc"],
            "final_test_best_aocc": dynagen["final_test_best_aocc"],
            "final_test_median_aocc": dynagen["final_test_median_aocc"],
            "final_test_mean_final_error": dynagen["final_test_mean_final_error"],
            "per_function_test_mean": dg_mean,
        },
        "llamea": {
            "name": llamea["name"],
            "exp_dir": llamea["exp_dir"],
            "candidates_total": llamea["candidates_total"],
            "candidates_valid": llamea["candidates_valid"],
            "best_train_aocc": llamea["best_train_aocc"],
            "best_train_id": llamea["best_train_id"],
            "final_selected_id": llamea["final_selected_id"],
            "final_train_mean_aocc": llamea["final_train_mean_aocc"],
            "final_test_mean_aocc": llamea["final_test_mean_aocc"],
            "per_function_test_mean": la_mean,
        },
        "per_function": per_function,
        "head_to_head": {
            "dynagen_wins": dg_wins,
            "llamea_wins": la_wins,
            "ties": ties,
            "compared": dg_wins + la_wins + ties,
        },
    }


def write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    dg = summary["dynagen"]
    la = summary["llamea"]
    h2h = summary["head_to_head"]

    def fmt(x: Any, digits: int = 4) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        if isinstance(x, float):
            return f"{x:.{digits}f}"
        return str(x)

    lines: list[str] = []
    lines.append("# BBOB comparison: DynaGen vs LLaMEA\n")
    lines.append("## Runs\n")
    lines.append(f"- **DynaGen**  `{dg['run_dir']}`")
    lines.append(f"- **LLaMEA**   `{la['exp_dir']}`\n")

    lines.append("## Summary\n")
    lines.append("| Metric | DynaGen | LLaMEA |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Candidates evaluated (total) | {dg['candidates_total']} | {la['candidates_total']} |"
    )
    lines.append(
        f"| Candidates evaluated (valid) | {dg['candidates_valid']} | {la['candidates_valid']} |"
    )
    lines.append(
        f"| Best train AOCC (any candidate) | {fmt(dg['best_train_aocc'])} | {fmt(la['best_train_aocc'])} |"
    )
    lines.append(
        f"| Final selected candidate id | {dg['final_selected_id']} | {la['final_selected_id']} |"
    )
    lines.append(
        f"| Final selected: test AOCC (mean) | {fmt(dg['final_test_mean_aocc'])} | {fmt(la['final_test_mean_aocc'])} |"
    )
    lines.append(
        f"| Final selected: per-function test mean | {fmt(dg['per_function_test_mean'])} | {fmt(la['per_function_test_mean'])} |"
    )
    lines.append(
        f"| Final selected: best single AOCC | {fmt(dg['final_test_best_aocc'])} | n/a |"
    )
    lines.append(
        f"| Final selected: median test AOCC | {fmt(dg['final_test_median_aocc'])} | n/a |"
    )
    lines.append(
        f"| Final selected: mean final error | {fmt(dg['final_test_mean_final_error'])} | n/a |"
    )
    lines.append("")

    lines.append("## Per-function head-to-head (test pool)\n")
    lines.append(
        f"DynaGen wins on **{h2h['dynagen_wins']}/{h2h['compared']}** functions, "
        f"LLaMEA wins on **{h2h['llamea_wins']}/{h2h['compared']}**, "
        f"ties: {h2h['ties']}.\n"
    )
    lines.append("| f | DynaGen AOCC | LLaMEA AOCC | Δ (DG-LA) | Winner |")
    lines.append("|---:|---:|---:|---:|:--|")
    for row in summary["per_function"]:
        lines.append(
            "| f{fn} | {dg} | {la} | {delta} | {w} |".format(
                fn=row["function"],
                dg=fmt(row["dynagen_aocc"]),
                la=fmt(row["llamea_aocc"]),
                delta=fmt(row["delta_dynagen_minus_llamea"]),
                w=row["winner"] or "n/a",
            )
        )
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- AOCC = Area Over the Convergence Curve, normalised to [0, 1] using "
        "lower bound 1e-8 and upper bound 100 (IOH default).  Higher is better."
    )
    lines.append(
        "- DynaGen test pool: 24 functions × 2 test instances × 1 seed = 48 runs. "
        "LLaMEA test pool: 24 functions × 2 test instances × 3 reps = 144 runs "
        "(averaged per function for the head-to-head)."
    )
    lines.append(
        "- DynaGen final selection is the algorithm chosen by the harness "
        "(see test_result.json); LLaMEA final selection is the highest-train-"
        "fitness candidate stored in solutions.jsonl."
    )
    out_path.write_text("\n".join(lines))


def write_convergence_csv(
    dynagen: dict[str, Any], llamea: dict[str, Any], out_path: Path
) -> None:
    """Write best-so-far AOCC per evaluation index for both runs (long format)."""
    dg_curve = dynagen["train_best_so_far_curve"]
    la_curve = llamea["train_best_so_far_curve"]
    n = max(len(dg_curve), len(la_curve))

    def fmt_val(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return ""
        return f"{float(v):.6f}"

    lines = ["eval_index,dynagen_best_so_far_aocc,llamea_best_so_far_aocc"]
    for i in range(n):
        dg = dg_curve[i] if i < len(dg_curve) else None
        la = la_curve[i] if i < len(la_curve) else None
        lines.append(f"{i + 1},{fmt_val(dg)},{fmt_val(la)}")
    out_path.write_text("\n".join(lines) + "\n")


def write_per_function_csv(summary: dict[str, Any], out_path: Path) -> None:
    lines = ["function,dynagen_aocc,llamea_aocc,delta,winner"]
    for row in summary["per_function"]:
        lines.append(
            "{fn},{dg},{la},{delta},{w}".format(
                fn=row["function"],
                dg="" if row["dynagen_aocc"] is None else f"{row['dynagen_aocc']:.6f}",
                la="" if row["llamea_aocc"] is None else f"{row['llamea_aocc']:.6f}",
                delta=(
                    ""
                    if row["delta_dynagen_minus_llamea"] is None
                    else f"{row['delta_dynagen_minus_llamea']:.6f}"
                ),
                w=row["winner"] or "",
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dynagen",
        default=DEFAULT_DYNAGEN,
        help="Path to a DynaGen BBOB run directory.",
    )
    parser.add_argument(
        "--llamea",
        default=DEFAULT_LLAMEA,
        help="Path to a LLaMEA experiment directory.",
    )
    parser.add_argument(
        "--out",
        default="bbob_compare_out",
        help="Output directory for plots and summary files.",
    )
    args = parser.parse_args()

    dyna_dir = Path(args.dynagen).expanduser().resolve()
    llamea_dir = Path(args.llamea).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dynagen = load_dynagen(dyna_dir)
    llamea = load_llamea(llamea_dir)
    summary = build_summary(dynagen, llamea)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, out_dir / "summary.md")
    write_per_function_csv(summary, out_dir / "per_function_aocc.csv")
    write_convergence_csv(dynagen, llamea, out_dir / "convergence.csv")
    plot_convergence(dynagen, llamea, out_dir / "convergence.png")
    plot_per_function(dynagen, llamea, out_dir / "per_function_aocc.png")

    # Console summary.
    dg = summary["dynagen"]
    la = summary["llamea"]
    h2h = summary["head_to_head"]
    print(f"DynaGen run : {dg['name']}")
    print(
        f"  candidates: {dg['candidates_valid']}/{dg['candidates_total']} valid  "
        f"best train AOCC: {dg['best_train_aocc']:.4f}  "
        f"final test AOCC: {dg['final_test_mean_aocc']:.4f}"
    )
    print(f"LLaMEA  exp : {la['name']}")
    print(
        f"  candidates: {la['candidates_valid']}/{la['candidates_total']} valid  "
        f"best train AOCC: {la['best_train_aocc']:.4f}  "
        f"final test AOCC: "
        + (
            "n/a"
            if la["final_test_mean_aocc"] is None
            else f"{la['final_test_mean_aocc']:.4f}"
        )
    )
    print(
        f"Per-function (test) head-to-head: DynaGen {h2h['dynagen_wins']}  "
        f"LLaMEA {h2h['llamea_wins']}  ties {h2h['ties']}  "
        f"(of {h2h['compared']} functions)"
    )
    print(f"\nWrote results to: {out_dir}")


if __name__ == "__main__":
    main()

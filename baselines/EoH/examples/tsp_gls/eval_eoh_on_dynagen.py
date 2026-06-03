#!/usr/bin/env python3
"""Evaluate EoH best candidates on DynaGen TSP test instances (parallel).

Reads samples_best.json from each results_20260531_* run, executes the
GLS black-box (NN + 2-opt + LLM-designed update_edge_distance) on every
DynaGen TSPLIB test instance, and prints mean gap % per instance.
"""

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# GLS primitives (copied from prob.py to stay self-contained)
# ---------------------------------------------------------------------------


def _tour_cost(tour, dist):
    return sum(dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1))


def _nearest_neighbour(dist, start=0):
    n = len(dist)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    for _ in range(n - 1):
        cur = tour[-1]
        nxt = min((j for j in range(n) if not visited[j]), key=lambda j: dist[cur, j])
        tour.append(nxt)
        visited[nxt] = True
    tour.append(start)
    return tour


def _two_opt(tour, dist):
    tour = list(tour)
    n = len(tour) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if (
                    dist[tour[i - 1], tour[j]] + dist[tour[i], tour[j + 1]]
                    < dist[tour[i - 1], tour[i]] + dist[tour[j], tour[j + 1]] - 1e-10
                ):
                    tour[i : j + 1] = tour[i : j + 1][::-1]
                    improved = True
    return tour


# ---------------------------------------------------------------------------
# GLS runner (one per worker process)
# ---------------------------------------------------------------------------


def _gls(dist, update_fn, iter_max=200, time_max=10.0):
    n = len(dist)
    tour = _nearest_neighbour(dist)
    tour = _two_opt(tour, dist)
    best_cost = _tour_cost(tour, dist)
    best_tour = tour[:]
    edge_n_used = np.zeros((n, n))

    t_end = time.time() + time_max
    for _ in range(iter_max):
        if time.time() > t_end:
            break
        aug = update_fn(
            dist.copy(),
            np.array(tour[:-1], dtype=int),
            edge_n_used.copy(),
        )
        aug = np.asarray(aug, dtype=float)
        aug = (aug + aug.T) / 2
        np.maximum(aug, 0, out=aug)
        gain = aug - dist
        np.fill_diagonal(gain, -np.inf)
        for _ in range(5):
            u, v = np.unravel_index(int(np.argmax(gain)), gain.shape)
            edge_n_used[u, v] += 1
            edge_n_used[v, u] += 1
            gain[u, v] = gain[v, u] = -np.inf
        tour = _two_opt(best_tour[:], aug)
        cost = _tour_cost(tour, dist)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]
    return best_cost


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------


def _eval_single(args):
    candidate_code, dist, optimal, iter_max, time_max = args
    ns = {"np": np, "__builtins__": __builtins__}
    exec(candidate_code, ns)
    update_fn = ns["update_edge_distance"]
    tour_length = _gls(dist, update_fn, iter_max=iter_max, time_max=time_max)
    gap = (tour_length - optimal) / optimal * 100 if optimal else None
    return tour_length, gap


# ---------------------------------------------------------------------------
# Instance loading (TSPLIB via DynaGen parser)
# ---------------------------------------------------------------------------


def load_tsplib(path):
    root = Path(__file__).resolve().parents[4]  # DynaGen project root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dynagen.domain.tsp_parser import load_tsplib_file

    return load_tsplib_file(path)


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------


def load_candidates(results_base):
    pattern = os.path.join(results_base, "results_20260531_*_pops5_gens20")
    dirs = sorted(glob.glob(pattern))
    candidates = []
    for d in dirs:
        best_path = os.path.join(d, "results", "samples", "samples_best.json")
        if not os.path.exists(best_path):
            continue
        with open(best_path) as f:
            data = json.load(f)
        tag = os.path.basename(d).replace("results_", "").split("_pops")[0]
        candidates.append(
            {
                "tag": tag,
                "code": data["code"],
                "objective": data["objective"],
                "operator": data["operator"],
            }
        )
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[4]  # DynaGen project root
    results_base = str(base)
    test_dir = root / "data" / "tsp" / "test_instances"

    candidates = load_candidates(results_base)
    if not candidates:
        print("No candidates found.", file=sys.stderr)
        return 1

    instances = sorted(test_dir.glob("*.tsp"))
    if not instances:
        print(f"No .tsp files in {test_dir}", file=sys.stderr)
        return 1

    print(f"Loaded {len(candidates)} candidates, {len(instances)} test instances\n")

    tasks = []
    for c in candidates:
        for inst_path in instances:
            inst = load_tsplib(inst_path)
            tasks.append(
                (
                    c["code"],
                    inst.distance_matrix,
                    inst.optimal_length,
                    200,
                    10.0,
                    c["tag"],
                    inst.name,
                )
            )

    results = {}
    with ProcessPoolExecutor() as pool:
        future_map = {}
        for code, dist, opt, imax, tmax, ctag, iname in tasks:
            f = pool.submit(_eval_single, (code, dist, opt, imax, tmax))
            future_map[f] = (ctag, iname)

        for f in as_completed(future_map):
            ctag, iname = future_map[f]
            try:
                tl, gap = f.result()
            except Exception as e:
                tl, gap = None, None
            results[(ctag, iname)] = (tl, gap)

    # -- pretty table --
    inst_names = [p.stem for p in instances]
    cand_tags = [c["tag"] for c in candidates]

    header = (
        f"{'Instance':<12}"
        + "".join(f"  {t:>12}" for t in cand_tags)
        + f"  {'Mean':>10}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    col_sums = {t: [] for t in cand_tags}
    for iname in inst_names:
        row = f"{iname:<12}"
        gaps = []
        for ctag in cand_tags:
            _, gap = results.get((ctag, iname), (None, None))
            if gap is not None:
                row += f"  {gap:>11.2f}%"
                col_sums[ctag].append(gap)
                gaps.append(gap)
            else:
                row += f"  {'ERR':>11}"
        mean_g = np.mean(gaps) if gaps else float("nan")
        row += f"  {mean_g:>9.2f}%"
        print(row)

    print(sep)
    row = f"{'Mean':<12}"
    for ctag in cand_tags:
        vals = col_sums[ctag]
        m = np.mean(vals) if vals else float("nan")
        row += f"  {m:>11.2f}%"
    overall = np.mean([v for v in col_sums.values() if v] or [float("nan")])
    row += f"  {overall:>9.2f}%"
    print(row)
    print()

    # -- summary --
    print("Candidate summary (training objective):")
    for c in candidates:
        print(f"  {c['tag']}  op={c['operator']}  train_obj={c['objective']:.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

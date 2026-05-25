import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=int)
        try:
            report_best_tour(tour)
        except:
            pass
        return tour
    rng = random.Random(seed)
    cand_size = 30 if n < 80 else 20
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        order = np.argsort(dists)
        order = order[order != i][:cand_size]
        candidates.append(set(order))
    max_restarts = min(10, max(1, budget // (n * 5)))
    base_per_restart = max(1, budget // max_restarts) if max_restarts > 0 else 0
    total_evals = 0
    best_tour = None
    best_dist = float('inf')
    
    for restart in range(max_restarts):
        if total_evals >= budget:
            break
        # Regret-2 construction
        start = rng.randrange(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_city = None
            best_regret = -1.0
            best_pos = None
            for city in unvisited:
                min_cost = float('inf')
                second_min = float('inf')
                best_idx = -1
                for idx in range(len(tour)):
                    if idx == len(tour) - 1:
                        cost = distance_matrix[tour[idx], city] + distance_matrix[city, tour[0]] - distance_matrix[tour[idx], tour[0]]
                    else:
                        cost = distance_matrix[tour[idx], city] + distance_matrix[city, tour[idx+1]] - distance_matrix[tour[idx], tour[idx+1]]
                    if cost < min_cost:
                        second_min = min_cost
                        min_cost = cost
                        best_idx = idx
                    elif cost < second_min:
                        second_min = cost
                regret = second_min - min_cost
                if regret > best_regret:
                    best_regret = regret
                    best_city = city
                    best_pos = best_idx
            tour.insert(best_pos + 1, best_city)
            unvisited.remove(best_city)
        tour = np.array(tour, dtype=int)
        # Compute initial distance
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            try:
                report_best_tour(best_tour)
            except:
                pass
        # 2-opt with candidate lists (first-improvement)
        pos = np.empty(n, dtype=int)
        for idx, node in enumerate(tour):
            pos[node] = idx
        improved = True
        restart_evals = 0
        allowed_moves = base_per_restart
        early_improvement = False
        while restart_evals < allowed_moves and total_evals < budget and improved:
            improved = False
            # generate candidate pairs
            pairs = []
            for i in range(n):
                a = tour[i]
                for b in candidates[a]:
                    j = pos[b]
                    if j <= i:
                        continue
                    if (i+1)%n == j or (j+1)%n == i:
                        continue
                    pairs.append((i, j))
            rng.shuffle(pairs)
            for i, j in pairs:
                if total_evals >= budget or restart_evals >= allowed_moves:
                    break
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                delta = new - old
                total_evals += 1
                restart_evals += 1
                if delta < -1e-12:
                    # apply 2-opt
                    if i < j:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    else:
                        segment = np.concatenate((tour[i+1:], tour[:j+1]))
                        segment = segment[::-1]
                        tour[i+1:] = segment[:n-i-1]
                        tour[:j+1] = segment[n-i-1:]
                    # update positions
                    for idx in range(n):
                        pos[tour[idx]] = idx
                    dist += delta
                    if dist < best_dist - 1e-12:
                        best_dist = dist
                        best_tour = tour.copy()
                        try:
                            report_best_tour(best_tour)
                        except:
                            pass
                        # check early improvement
                        if not early_improvement and restart_evals <= base_per_restart // 2:
                            early_improvement = True
                            allowed_moves = min(int(1.5 * base_per_restart), budget - total_evals + restart_evals)
                    improved = True
                    break
        # Ensure we consume some budget even if no improvement? Already considered.
    return best_tour
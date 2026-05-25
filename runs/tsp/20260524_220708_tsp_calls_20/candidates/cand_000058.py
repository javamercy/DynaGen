import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)

    def randomized_regret_construction(start):
        tour = [start]
        unvisited = set(range(n)) - {start}
        while unvisited:
            candidates = []
            for city in unvisited:
                best_cost = np.inf
                second_best = np.inf
                best_pos = -1
                m = len(tour)
                for i in range(m):
                    prev = tour[i]
                    nxt = tour[(i+1) % m]
                    inc = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                    if inc < best_cost:
                        second_best = best_cost
                        best_cost = inc
                        best_pos = i+1
                    elif inc < second_best:
                        second_best = inc
                regret = second_best - best_cost if second_best != np.inf else best_cost
                candidates.append((city, regret, best_pos))
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_k = min(3, len(candidates))
            idx = rng.integers(top_k)
            chosen, _, pos = candidates[idx]
            tour.insert(pos, chosen)
            unvisited.remove(chosen)
        return np.array(tour, dtype=np.int64)

    start = rng.integers(n)
    tour = randomized_regret_construction(start)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)

    ops = 0
    pass_no_improve = 0
    max_passes = 3
    cur_dist = best_dist

    while ops < budget:
        improved_pass = False
        for i in range(n-1):
            for j in range(i+2, n):
                ops += 1
                if ops >= budget:
                    break
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                if new < old - 1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    cur_dist = 0.0
                    for k in range(n):
                        cur_dist += distance_matrix[tour[k], tour[(k+1)%n]]
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved_pass = True
                    break
            if ops >= budget:
                break
            if improved_pass:
                break
        if ops >= budget:
            break
        if improved_pass:
            pass_no_improve = 0
        else:
            pass_no_improve += 1
            if pass_no_improve >= max_passes:
                # perturbation: random 2-opt reversal
                i = rng.integers(n)
                j = rng.integers(n)
                while i == j:
                    j = rng.integers(n)
                start_idx = min(i, j)
                end_idx = max(i, j)
                while start_idx == 0 and end_idx == n-1:
                    i = rng.integers(n)
                    j = rng.integers(n)
                    while i == j:
                        j = rng.integers(n)
                    start_idx = min(i, j)
                    end_idx = max(i, j)
                tour[start_idx+1:end_idx+1] = tour[start_idx+1:end_idx+1][::-1]
                cur_dist = 0.0
                for k in range(n):
                    cur_dist += distance_matrix[tour[k], tour[(k+1)%n]]
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                pass_no_improve = 0
    return best_tour
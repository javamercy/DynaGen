import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)
    # Regret-based construction
    start = rng.integers(n)
    tour = [start]
    unvisited = set(range(n)) - {start}
    while unvisited:
        best_insert = {}
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
            best_insert[city] = (regret, best_pos, best_cost)
        chosen = max(unvisited, key=lambda c: best_insert[c][0])
        pos = best_insert[chosen][1]
        tour.insert(pos, chosen)
        unvisited.remove(chosen)
    tour = np.array(tour, dtype=np.int64)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)

    def compute_dist(t):
        return distance_matrix[t[-1], t[0]] + sum(distance_matrix[t[i], t[i+1]] for i in range(n-1))

    cur_dist = best_dist
    ops = 0
    local_opt = False
    while ops < budget:
        improved = False
        for i in range(n-1):
            if ops >= budget:
                break
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                if new < old - 1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    cur_dist += (new - old)
                    improved = True
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
        if not improved:
            # Double-bridge perturbation
            if ops >= budget:
                break
            # pick 4 distinct cut points
            if n < 8:
                break
            cuts = sorted(rng.choice(range(1, n-1), size=4, replace=False))
            a, b, c, d = cuts
            # rebuild tour: segments [0:a], [b:c], [a:b], [c:d], [d:]
            new_tour = np.concatenate([tour[:a], tour[b:c], tour[a:b], tour[c:d], tour[d:]])
            # ensure it's a valid permutation
            tour = new_tour
            cur_dist = compute_dist(tour)
            # note: we do not report this as best unless better
            # continue local search
        else:
            local_opt = False
    return best_tour
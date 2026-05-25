import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)
    dist = distance_matrix

    def regret_construction(start):
        tour = [start]
        unvisited = set(range(n)) - {start}
        while unvisited:
            best_data = {}
            for city in unvisited:
                best_cost = np.inf
                second_best = np.inf
                best_pos = -1
                m = len(tour)
                for i in range(m):
                    prev = tour[i]
                    nxt = tour[(i+1)%m]
                    inc = dist[prev, city] + dist[city, nxt] - dist[prev, nxt]
                    if inc < best_cost:
                        second_best = best_cost
                        best_cost = inc
                        best_pos = i+1
                    elif inc < second_best:
                        second_best = inc
                regret = second_best - best_cost if second_best != np.inf else best_cost
                best_data[city] = (regret, best_pos, best_cost)
            chosen = max(unvisited, key=lambda c: best_data[c][0])
            pos = best_data[chosen][1]
            tour.insert(pos, chosen)
            unvisited.remove(chosen)
        return np.array(tour, dtype=np.int64)

    start = rng.integers(n)
    tour = regret_construction(start)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += dist[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    cur_dist = best_dist
    ops = 0
    stall = 0
    stall_limit = 3

    while ops < budget:
        # First-improvement 2-opt pass
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                old = dist[a,b] + dist[c,d]
                new = dist[a,c] + dist[b,d]
                if new < old - 1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    cur_dist += new - old
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if ops >= budget:
                break
            if improved:
                break
        if ops >= budget:
            break
        if improved:
            stall = 0
        else:
            stall += 1
            if stall >= stall_limit:
                # Double-bridge perturbation
                if ops + n >= budget:
                    break
                # Generate 4 random distinct indices sorted
                idx = sorted(rng.choice(n, 4, replace=False))
                a, b, c, d = idx
                # Build new tour
                new_tour = np.zeros(n, dtype=np.int64)
                pos = 0
                for k in range(a+1):
                    new_tour[pos] = tour[k]
                    pos += 1
                for k in range(c+1, d+1):
                    new_tour[pos] = tour[k]
                    pos += 1
                for k in range(a+1, b+1):
                    new_tour[pos] = tour[k]
                    pos += 1
                for k in range(d+1, n):
                    new_tour[pos] = tour[k]
                    pos += 1
                for k in range(b+1, c+1):
                    new_tour[pos] = tour[k]
                    pos += 1
                tour = new_tour
                # Recompute full distance
                ops += n
                cur_dist = 0.0
                for i in range(n):
                    cur_dist += dist[tour[i], tour[(i+1)%n]]
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                stall = 0
    return best_tour
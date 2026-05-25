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

    # 2-opt with adaptive restart and double-bridge perturbation
    ops = 0
    improved = True
    stagnation = 0
    threshold = 1
    while ops < budget:
        if not improved:
            stagnation += 1
            # Adaptive threshold: increases with budget consumption
            threshold = 1 + int((ops / budget) * 9) if budget > 0 else 10
            if stagnation > threshold:
                # Double-bridge perturbation instead of full restart
                if ops < budget:
                    ops += 1  # count perturbation as one operation
                    # perform double-bridge
                    i = rng.integers(0, n-3)
                    j = rng.integers(i+1, n-2)
                    k = rng.integers(j+1, n-1)
                    l = rng.integers(k+1, n)
                    # Segments: S1=tour[:i+1], S2=tour[i+1:j+1], S3=tour[j+1:k+1], S4=tour[k+1:l+1], S5=tour[l+1:]
                    new_tour = np.concatenate([tour[:i+1], tour[j+1:k+1], tour[i+1:j+1], tour[k+1:l+1], tour[l+1:]])
                    # Ensure valid (should be length n)
                    if len(new_tour) != n:
                        # fallback: keep current
                        new_tour = tour.copy()
                    # compute new distance
                    cur_dist = 0.0
                    for t in range(n):
                        cur_dist += distance_matrix[new_tour[t], new_tour[(t+1)%n]]
                    tour = new_tour
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    stagnation = 0
                continue
            # else continue scanning but improved stays False
        else:
            stagnation = 0
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
                    improved = True
                    cur_dist = best_dist + (new - old)
                    if cur_dist < best_dist - 1e-12:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour
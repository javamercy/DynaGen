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
    
    # 2-opt with adaptive restart and double-bridge
    ops = 0
    improved = True
    stagnation = 0
    threshold = 1
    double_bridge_used = False
    while ops < budget:
        if not improved:
            stagnation += 1
            # Adaptive threshold: increases with budget consumption
            threshold = 1 + int((ops / budget) * 9) if budget > 0 else 10
            if stagnation > threshold:
                if not double_bridge_used:
                    # Double-bridge perturbation
                    cuts = sorted(rng.choice(range(1, n), 4, replace=False))
                    slices = [tour[:cuts[0]], tour[cuts[0]:cuts[1]], tour[cuts[1]:cuts[2]], tour[cuts[2]:cuts[3]], tour[cuts[3]:]]
                    tour = np.concatenate([slices[0], slices[3], slices[1], slices[2], slices[4]])
                    # Update best if improved
                    cur_dist = 0.0
                    for i in range(n):
                        cur_dist += distance_matrix[tour[i], tour[(i+1)%n]]
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    double_bridge_used = True
                    improved = True
                    stagnation = 0
                    continue
                else:
                    # Restart with regret construction
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
                    double_bridge_used = False
                    improved = True
                    stagnation = 0
                    continue
        else:
            stagnation = 0
            double_bridge_used = False
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
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour
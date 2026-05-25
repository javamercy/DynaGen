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
        best_data = {}
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
            best_data[city] = (regret, best_pos, best_cost)
        chosen = max(unvisited, key=lambda c: best_data[c][0])
        pos = best_data[chosen][1]
        tour.insert(pos, chosen)
        unvisited.remove(chosen)
    tour = np.array(tour, dtype=np.int64)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)

    def segment_shift(tour):
        # Or-opt style: remove a contiguous segment and insert elsewhere
        length = rng.integers(2, max(2, n//4))
        start_idx = rng.integers(0, n - length)
        seg = tour[start_idx:start_idx+length].copy()
        # Remove segment
        rem = np.concatenate([tour[:start_idx], tour[start_idx+length:]])
        # Choose insertion point not at gap (avoid trivial)
        insert_pos = rng.integers(0, len(rem)+1)
        new_tour = np.concatenate([rem[:insert_pos], seg, rem[insert_pos:]])
        return new_tour

    ops = 0
    improved = True
    while ops < budget:
        if not improved:
            # Perturbation
            if ops >= budget:
                break
            ops += 1
            tour = segment_shift(tour)
            improved = True
            # After perturbation, try limited 3-opt
            cur_dist = 0.0
            for i in range(n):
                cur_dist += distance_matrix[tour[i], tour[(i+1)%n]]
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            # 3-opt single pass
            for i in range(n-2):
                if ops >= budget:
                    break
                for j in range(i+1, n-1):
                    if ops >= budget:
                        break
                    for k in range(j+1, n):
                        if ops >= budget:
                            break
                        ops += 1
                        # Original edges: a-b, c-d, e-f
                        a = tour[i]; b = tour[(i+1)%n]
                        c = tour[j]; d = tour[(j+1)%n]
                        e = tour[k]; f = tour[(k+1)%n]
                        # Try 2-opt like reversals? Actually 3-opt: check alternative connections
                        # We only check if reversing segment i+1..j improves (that's 2-opt)
                        # For true 3-opt, we'd need more combos, but limit to simple check
                        # Instead, we can try to swap order of segments: not implement full 3-opt
                        # To keep it simple, we skip full 3-opt and rely on 2-opt and perturbation
                        pass
            continue
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
                    cur_dist = 0.0
                    for k in range(n):
                        cur_dist += distance_matrix[tour[k], tour[(k+1)%n]]
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour
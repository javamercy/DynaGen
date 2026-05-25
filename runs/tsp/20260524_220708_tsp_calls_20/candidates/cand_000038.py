import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)

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
        return np.array(tour, dtype=np.int64)

    def compute_dist(tour):
        d = 0.0
        for i in range(n):
            d += distance_matrix[tour[i], tour[(i+1)%n]]
        return d

    start = rng.integers(n)
    tour = regret_construction(start)
    best_tour = tour.copy()
    best_dist = compute_dist(tour)
    report_best_tour(best_tour)

    ops = 0
    improved = True
    while ops < budget:
        if not improved:
            # segment shift perturbation
            L = rng.integers(1, n//2+1)
            i = rng.integers(0, n - L + 1)
            seg = tour[i:i+L].copy()
            rest = np.concatenate([tour[:i], tour[i+L:]])
            pos = rng.integers(0, len(rest)+1)
            tour = np.concatenate([rest[:pos], seg, rest[pos:]])
            ops += 1
            cur_dist = compute_dist(tour)
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            improved = True
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
                    cur_dist = compute_dist(tour)
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    # post-optimization: limited segment moves (3-opt like)
    max_post = min(10, budget - ops)
    for _ in range(max_post):
        if ops >= budget:
            break
        # choose random segment
        L = rng.integers(1, max(2, n//2+1))
        i = rng.integers(0, n - L + 1)
        seg = tour[i:i+L].copy()
        rest = np.concatenate([tour[:i], tour[i+L:]])
        # evaluate best insertion position
        best_inc = 0.0
        best_pos = -1
        m_rest = len(rest)
        # precompute distances around rest for delta
        for p in range(m_rest+1):
            left = rest[p-1] if p > 0 else rest[-1]
            right = rest[p] if p < m_rest else rest[0]
            # cost of removing edges around seg in original? Actually we compute delta using rest and seg
            # Simplified: compute full distance after insertion? Not efficient. Use delta.
            # We'll just compute full distance for simplicity (small n).
            candidate = np.concatenate([rest[:p], seg, rest[p:]])
            d = compute_dist(candidate)
            inc = d - best_dist
            if inc < best_inc:
                best_inc = inc
                best_pos = p
        if best_inc < -1e-12:
            ops += 1
            tour = np.concatenate([rest[:best_pos], seg, rest[best_pos:]])
            best_dist += best_inc
            best_tour = tour.copy()
            report_best_tour(best_tour)
        else:
            ops += 1
    return best_tour
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
                best_cost, second_best, best_pos = np.inf, np.inf, -1
                m = len(tour)
                for i in range(m):
                    prev = tour[i]
                    nxt = tour[(i+1) % m]
                    inc = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                    if inc < best_cost:
                        second_best, best_cost, best_pos = best_cost, inc, i+1
                    elif inc < second_best:
                        second_best = inc
                regret = second_best - best_cost if second_best != np.inf else best_cost
                best_data[city] = (regret, best_pos, best_cost)
            chosen = max(unvisited, key=lambda c: best_data[c][0])
            pos = best_data[chosen][1]
            tour.insert(pos, chosen)
            unvisited.remove(chosen)
        return np.array(tour, dtype=np.int64)

    def segment_shift(tour):
        # randomly choose a segment [l, r] and a new insertion point p (not inside segment)
        l = rng.integers(0, n)
        seg_len = rng.integers(1, min(n-1, 6))  # small segment
        r = (l + seg_len) % n
        if r < l:
            segment = np.concatenate([tour[l:], tour[:r+1]])
            remainder = tour[r+1:l]
        else:
            segment = tour[l:r+1]
            remainder = np.concatenate([tour[r+1:], tour[:l]])
        # choose new position in remainder (excluding positions that would insert inside segment)
        p = rng.integers(0, len(remainder))
        new_tour = np.concatenate([remainder[:p], segment, remainder[p:]])
        return new_tour

    start = rng.integers(n)
    tour = regret_construction(start)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)

    ops = 0
    improved = True
    while ops < budget:
        if not improved:
            tour = segment_shift(tour)
            ops += 1
            cur_dist = 0.0
            for i in range(n):
                cur_dist += distance_matrix[tour[i], tour[(i+1)%n]]
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
    # limited 3-opt post-processing if budget remains
    if ops < budget:
        for _ in range(min(10, budget - ops)):
            ops += 1
            i, j, k = sorted(rng.choice(n, 3, replace=False))
            i, j, k = i % n, j % n, k % n
            # try all 4 possible reconnections (excluding original)
            # only consider the two that reverse segments (standard 3-opt moves)
            # simplified: try two reversed versions: (i+1..j reversed) and (j+1..k reversed) etc.
            # but for compactness, try a 2-opt move on a random segment
            a = rng.integers(0, n)
            b = (a + rng.integers(2, n-1)) % n
            old = distance_matrix[tour[a], tour[(a+1)%n]] + distance_matrix[tour[b], tour[(b+1)%n]]
            new = distance_matrix[tour[a], tour[b]] + distance_matrix[tour[(a+1)%n], tour[(b+1)%n]]
            if new < old - 1e-12:
                if a < b:
                    tour[a+1:b+1] = tour[a+1:b+1][::-1]
                else:
                    seg = np.concatenate([tour[a+1:], tour[:b+1]])
                    seg = seg[::-1]
                    tour[a+1:] = seg[:n-a-1]
                    tour[:b+1] = seg[n-a-1:]
                cur_dist = 0.0
                for kk in range(n):
                    cur_dist += distance_matrix[tour[kk], tour[(kk+1)%n]]
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
    return best_tour
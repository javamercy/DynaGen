import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    np.random.seed(seed)

    # nearest neighbor construction from random start
    start = np.random.randint(n)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour, dtype=np.int32)
    report_best_tour(tour.copy())

    # candidate lists: top 15 nearest neighbors (or n-1 if n<80)
    cand_size = min(15, n-1) if n >= 80 else n-1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        if cand_size < n - 1:
            idx = np.argpartition(dists, cand_size)[:cand_size]
            idx_sorted = idx[np.argsort(dists[idx])]
        else:
            idx_sorted = np.argsort(dists)[1:]
        candidates.append(idx_sorted)

    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1) % n]]
        return total

    best_tour = tour.copy()
    best_len = tour_length(best_tour)
    eps = 1e-12
    improved = True

    def apply_2opt(t, i, j):
        # reverse segment from i+1 to j (inclusive) handling wrap-around
        ip1 = (i + 1) % n
        if ip1 <= j:
            t[ip1:j+1] = t[ip1:j+1][::-1]
        else:
            segment = np.concatenate([t[ip1:], t[:j+1]])
            segment = segment[::-1]
            t[ip1:] = segment[:n-ip1]
            t[:j+1] = segment[n-ip1:]
        return t

    def double_bridge(t):
        if n < 8:
            return t
        positions = sorted(np.random.choice(range(n), 4, replace=False))
        a, b, c, d = positions
        seg1 = t[:a+1]
        seg2 = t[c+1:d+1]
        seg3 = t[b+1:c+1]
        seg4 = t[a+1:b+1]
        seg5 = t[d+1:]
        new_t = np.concatenate([seg1, seg2, seg3, seg4, seg5])
        return new_t.astype(np.int32)

    while budget > 0 and improved:
        improved = False
        best_delta = 0.0
        best_i = -1
        best_j = -1
        # scan all possible moves
        for i in range(n):
            if budget <= 0:
                break
            ip1 = (i + 1) % n
            a = tour[i]
            b = tour[ip1]
            for j_cand in candidates[b]:
                if budget <= 0:
                    break
                # find position j in tour
                j = -1
                for idx in range(n):
                    if tour[idx] == j_cand:
                        j = idx
                        break
                if j == -1:
                    continue
                # skip same edge, adjacent edges
                if j == i or j == ip1 or j == (i - 1) % n:
                    continue
                jp1 = (j + 1) % n
                c = tour[j]
                d = tour[jp1]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                budget -= 1
                if delta < best_delta - eps:
                    best_delta = delta
                    best_i = i
                    best_j = j
        if best_delta < -eps and best_i != -1:
            # apply best move
            tour = apply_2opt(tour, best_i, best_j)
            improved = True
            new_len = tour_length(tour)
            if new_len < best_len - eps:
                best_len = new_len
                best_tour = tour.copy()
                report_best_tour(best_tour.copy())
        elif not improved and budget > 0:
            # apply perturbation
            tour = double_bridge(tour)
            new_len = tour_length(tour)
            if new_len < best_len - eps:
                best_len = new_len
                best_tour = tour.copy()
                report_best_tour(best_tour.copy())
            improved = True  # continue searching after perturbation

    return best_tour
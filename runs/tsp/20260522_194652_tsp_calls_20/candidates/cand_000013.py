import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    np.random.seed(seed)
    # random initial tour
    tour = np.random.permutation(n).astype(np.int32)
    report_best_tour(tour.copy())

    # candidate lists (top 15 nearest neighbors)
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

    # helper to compute tour length
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1)%n]]
        return total

    # position array for quick city lookup
    pos = np.empty(n, dtype=np.int32)
    for idx, city in enumerate(tour):
        pos[city] = idx

    best_tour = tour.copy()
    best_len = tour_length(best_tour)

    eps = 1e-12
    improved = True

    while budget > 0 and improved:
        improved = False
        best_delta = 0.0
        best_i = -1
        best_j = -1

        # best-improvement scan of all 2-opt moves
        for i in range(n):
            if budget <= 0:
                break
            ip1 = (i + 1) % n
            a = tour[i]
            b = tour[ip1]
            for j_cand in candidates[b]:
                if budget <= 0:
                    break
                j = pos[j_cand]
                # skip invalid edges
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

        if best_delta < -eps:
            # apply best move
            i = best_i
            j = best_j
            ip1 = (i + 1) % n
            # reverse segment from ip1 to j
            if ip1 <= j:
                tour[ip1:j+1] = tour[ip1:j+1][::-1]
                # update pos for reversed segment
                seg = list(range(ip1, j+1))
                for idx in seg:
                    pos[tour[idx]] = idx
            else:
                segment = np.concatenate([tour[ip1:], tour[:j+1]])
                segment = segment[::-1]
                # update pos for first part
                for idx in range(ip1, n):
                    pos[tour[idx]] = -1  # temporary
                tour[ip1:] = segment[:n-ip1]
                for idx in range(ip1, n):
                    pos[tour[idx]] = idx
                tour[:j+1] = segment[n-ip1:]
                for idx in range(j+1):
                    pos[tour[idx]] = idx

            new_len = tour_length(tour)
            if new_len < best_len - eps:
                best_len = new_len
                best_tour = tour.copy()
                report_best_tour(best_tour.copy())
            improved = True
        else:
            # local optimum, apply double-bridge perturbation if budget remains
            if budget > 0:
                # pick three random cut points
                i1 = np.random.randint(0, n-3)
                i2 = np.random.randint(i1+1, n-2)
                i3 = np.random.randint(i2+1, n-1)
                new_tour = np.concatenate([
                    tour[:i1+1],
                    tour[i2+1:i3+1],
                    tour[i1+1:i2+1],
                    tour[i3+1:]
                ])
                tour = new_tour
                # update pos
                for idx, city in enumerate(tour):
                    pos[city] = idx
                budget -= 1  # cost of perturbation
                new_len = tour_length(tour)
                if new_len < best_len - eps:
                    best_len = new_len
                    best_tour = tour.copy()
                    report_best_tour(best_tour.copy())
                improved = True

    return best_tour
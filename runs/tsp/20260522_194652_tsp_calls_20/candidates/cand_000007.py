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

    # candidate lists: nearest 15 (or n-1 if n < 80)
    cand_size = min(15, n-1) if n >= 80 else n-1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        sorted_idx = np.argsort(dists)
        # exclude self (index 0 is self)
        candidates.append(sorted_idx[1:cand_size+1])

    # position map: city -> index in tour
    pos = np.zeros(n, dtype=np.int32)
    for idx, city in enumerate(tour):
        pos[city] = idx

    def tour_length(t):
        total = 0.0
        for k in range(n):
            total += distance_matrix[t[k], t[(k+1) % n]]
        return total

    best_tour = tour.copy()
    best_len = tour_length(tour)
    eps = 1e-12

    while budget > 0:
        # best-improvement pass
        best_delta = 0.0
        best_i = -1
        best_j = -1
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
                # skip self and adjacent edges
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
            # apply best 2-opt move
            i = best_i
            j = best_j
            ip1 = (i + 1) % n
            jp1 = (j + 1) % n  # not used directly
            # reverse segment from ip1 to j
            if ip1 <= j:
                tour[ip1:j+1] = tour[ip1:j+1][::-1]
                for idx in range(ip1, j+1):
                    city = tour[idx]
                    pos[city] = idx
            else:
                # wrap-around case
                segment = np.concatenate([tour[ip1:], tour[:j+1]])
                segment = segment[::-1]
                tour[ip1:] = segment[:n-ip1]
                tour[:j+1] = segment[n-ip1:]
                for idx, city in enumerate(tour):
                    pos[city] = idx
            new_len = tour_length(tour)
            if new_len < best_len - eps:
                best_len = new_len
                best_tour = tour.copy()
                report_best_tour(best_tour.copy())
        else:
            # no improvement: double-bridge perturbation
            if n < 4:
                break
            cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
            a, b, c = cuts
            # segments: [0:a], [a:b], [b:c], [c:n] -> reorder: [0:a] + [b:c] + [a:b] + [c:n]
            new_tour = np.concatenate([tour[:a], tour[b:c], tour[a:b], tour[c:]])
            tour = new_tour
            # update pos
            for idx, city in enumerate(tour):
                pos[city] = idx
            # continue searching

    return best_tour

# The following is required by the problem specification but not part of the returned code.
# def report_best_tour(tour):
#     pass
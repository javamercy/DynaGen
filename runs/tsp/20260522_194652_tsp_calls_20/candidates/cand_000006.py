import numpy as np


def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    # initialize RNG
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
            # use argpartition to get nearest indices
            idx = np.argpartition(dists, cand_size)[:cand_size]
            # sort within subset by distance
            idx_sorted = idx[np.argsort(dists[idx])]
        else:
            idx_sorted = np.argsort(dists)[1:]  # skip self
        candidates.append(idx_sorted)

    # helper for tour length (optional, used for reporting)
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1) % n]]
        return total

    best_tour = tour.copy()
    best_len = tour_length(best_tour)
    improved = True
    eps = 1e-12

    # main 2-opt loop
    while budget > 0 and improved:
        improved = False
        # iterate over all edges (i, i+1)
        for i in range(n):
            if budget <= 0:
                break
            ip1 = (i + 1) % n
            a = tour[i]
            b = tour[ip1]
            # consider candidates of b as potential j
            for j_cand in candidates[b]:
                if budget <= 0:
                    break
                # find position of j_cand in tour (j)
                # since we have limited candidates, linear search is okay
                # but we need to be careful with wrap-around; we search all positions
                # Could precompute positions but overhead not needed
                j = -1
                for idx in range(n):
                    if tour[idx] == j_cand:
                        j = idx
                        break
                if j == -1:
                    continue
                # avoid same edge and adjacent edges
                if j == i or j == ip1 or j == (i - 1) % n:
                    continue
                jp1 = (j + 1) % n
                c = tour[j]
                d = tour[jp1]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                budget -= 1
                if delta < -eps:
                    # apply 2-opt: reverse segment from ip1 to j
                    if ip1 <= j:
                        tour[ip1:j+1] = tour[ip1:j+1][::-1]
                    else:
                        # wrap-around: concatenate two parts, reverse, replace
                        segment = np.concatenate([tour[ip1:], tour[:j+1]])
                        segment = segment[::-1]
                        tour[ip1:] = segment[:n-ip1]
                        tour[:j+1] = segment[n-ip1:]
                    improved = True
                    # compute new length once for reporting
                    new_len = tour_length(tour)
                    if new_len < best_len - eps:
                        best_len = new_len
                        best_tour = tour.copy()
                        report_best_tour(best_tour.copy())
                    break  # first improvement
        # if no improvement found, exit

    return best_tour
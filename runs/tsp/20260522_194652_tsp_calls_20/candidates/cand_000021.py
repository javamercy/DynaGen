import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    np.random.seed(seed)

    # candidate lists
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
            total += distance_matrix[t[i], t[(i+1)%n]]
        return total

    def nearest_neighbor_tour():
        start = np.random.randint(n)
        unvisited = set(range(n))
        unvisited.remove(start)
        tour_list = [start]
        current = start
        while unvisited:
            next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour_list.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        return np.array(tour_list, dtype=np.int32)

    def double_bridge(t):
        # choose 4 random breakpoints in sorted order
        positions = sorted(np.random.choice(range(1, n), size=4, replace=False))
        a, b, c, d = positions[0], positions[1], positions[2], positions[3]
        # segments: [0:a], [a:b], [b:c], [c:d], [d:n]
        # reorder: [0:a], [c:d], [b:c], [a:b], [d:n]
        t_new = np.concatenate([t[:a], t[c:d], t[b:c], t[a:b], t[d:]])
        return t_new

    # initial tour
    tour = nearest_neighbor_tour()
    report_best_tour(tour.copy())
    best_tour = tour.copy()
    best_len = tour_length(best_tour)
    eps = 1e-12

    while budget > 0:
        improved = False
        # 2-opt first-improvement
        for i in range(n):
            if budget <= 0:
                break
            ip1 = (i + 1) % n
            a = tour[i]
            b = tour[ip1]
            for j_cand in candidates[b]:
                if budget <= 0:
                    break
                # find position of j_cand
                j = -1
                for idx in range(n):
                    if tour[idx] == j_cand:
                        j = idx
                        break
                if j == -1:
                    continue
                if j == i or j == ip1 or j == (i - 1) % n:
                    continue
                jp1 = (j + 1) % n
                c = tour[j]
                d = tour[jp1]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                budget -= 1
                if delta < -eps:
                    # apply 2-opt reverse
                    if ip1 <= j:
                        tour[ip1:j+1] = tour[ip1:j+1][::-1]
                    else:
                        segment = np.concatenate([tour[ip1:], tour[:j+1]])
                        segment = segment[::-1]
                        tour[ip1:] = segment[:n-ip1]
                        tour[:j+1] = segment[n-ip1:]
                    improved = True
                    new_len = tour_length(tour)
                    if new_len < best_len - eps:
                        best_len = new_len
                        best_tour = tour.copy()
                        report_best_tour(best_tour.copy())
                    break  # first improvement
        if not improved and budget > 0:
            # apply double-bridge perturbation
            budget -= 1  # charge for perturbation
            tour = double_bridge(tour)
            # note: perturbation may increase length, no report here
        elif not improved:
            break
    return best_tour
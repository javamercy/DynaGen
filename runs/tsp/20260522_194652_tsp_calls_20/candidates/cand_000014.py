import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    np.random.seed(seed)

    # nearest neighbor initial tour from random start
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

    best_tour = tour.copy()
    best_len = tour_length(best_tour)
    current_tour = tour.copy()
    current_len = best_len
    improved = True
    eps = 1e-12

    while budget > 0:
        # first-improvement 2-opt with candidate lists
        improved = False
        for i in range(n):
            if budget <= 0:
                break
            ip1 = (i + 1) % n
            a = current_tour[i]
            b = current_tour[ip1]
            for j_cand in candidates[b]:
                if budget <= 0:
                    break
                # find position of j_cand
                j = -1
                for idx in range(n):
                    if current_tour[idx] == j_cand:
                        j = idx
                        break
                if j == -1:
                    continue
                if j == i or j == ip1 or j == (i - 1) % n:
                    continue
                jp1 = (j + 1) % n
                c = current_tour[j]
                d = current_tour[jp1]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                budget -= 1
                if delta < -eps:
                    # reverse segment from ip1 to j
                    if ip1 <= j:
                        current_tour[ip1:j+1] = current_tour[ip1:j+1][::-1]
                    else:
                        segment = np.concatenate([current_tour[ip1:], current_tour[:j+1]])
                        segment = segment[::-1]
                        current_tour[ip1:] = segment[:n-ip1]
                        current_tour[:j+1] = segment[n-ip1:]
                    improved = True
                    current_len = tour_length(current_tour)
                    if current_len < best_len - eps:
                        best_len = current_len
                        best_tour = current_tour.copy()
                        report_best_tour(best_tour.copy())
                    break  # first improvement
        if not improved or budget <= 0:
            if budget <= 0:
                break
            # random 2-opt perturbation: reverse a random segment
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i > j:
                i, j = j, i
            # ensure segment has at least 2 cities (j > i+1)
            if j - i < 2:
                j = (j + 2) % n
                if i > j:
                    i, j = j, i
            # reverse from i+1 to j
            if i+1 <= j:
                current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
            else:
                segment = np.concatenate([current_tour[i+1:], current_tour[:j+1]])
                segment = segment[::-1]
                current_tour[i+1:] = segment[:n-i-1]
                current_tour[:j+1] = segment[n-i-1:]
            budget -= 1
    return best_tour
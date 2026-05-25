import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour

    np.random.seed(seed)

    # candidate lists (first 15 or n-1 nearest neighbors)
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

    def regret_construction():
        # start with a random city
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n)) - {start}
        while unvisited:
            best_city = None
            best_pos = -1
            best_regret = -np.inf
            best_second = np.inf
            best_first = np.inf
            for city in unvisited:
                first = np.inf
                second = np.inf
                best_p = -1
                for i in range(len(tour)):
                    j = (i + 1) % len(tour)
                    inc = distance_matrix[tour[i], city] + distance_matrix[city, tour[j]] - distance_matrix[tour[i], tour[j]]
                    if inc < first:
                        second = first
                        first = inc
                        best_p = i + 1
                    elif inc < second:
                        second = inc
                regret = second - first
                if regret > best_regret:
                    best_regret = regret
                    best_city = city
                    best_pos = best_p
                    best_first = first
                    best_second = second
            # insert best_city at best_pos
            tour.insert(best_pos, best_city)
            unvisited.remove(best_city)
        return np.array(tour, dtype=np.int32)

    # initial tour
    tour = regret_construction()
    report_best_tour(tour.copy())
    best_tour = tour.copy()
    best_len = tour_length(best_tour)
    eps = 1e-12

    while budget > 0:
        improved = False
        # first-improvement 2-opt over candidate lists
        for i in range(n):
            if budget <= 0:
                break
            ip1 = (i + 1) % n
            a = tour[i]
            b = tour[ip1]
            for j_cand in candidates[b]:
                if budget <= 0:
                    break
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
                    break
        if not improved and budget > 0:
            budget -= 1
            tour = regret_construction()
            new_len = tour_length(tour)
            if new_len < best_len - eps:
                best_len = new_len
                best_tour = tour.copy()
                report_best_tour(best_tour.copy())
        elif not improved:
            break
    return best_tour
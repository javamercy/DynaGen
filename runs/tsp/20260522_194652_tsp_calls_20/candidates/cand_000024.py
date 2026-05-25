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

    def random_insertion_tour():
        start = np.random.randint(n)
        second = np.random.choice([c for c in range(n) if c != start])
        tour = [start, second]
        unvisited = set(range(n)) - {start, second}
        for _ in range(n - 2):
            next_city = np.random.choice(list(unvisited))
            unvisited.remove(next_city)
            best_pos = -1
            best_increase = float('inf')
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                increase = distance_matrix[tour[i], next_city] + distance_matrix[next_city, tour[j]] - distance_matrix[tour[i], tour[j]]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1
            tour.insert(best_pos, next_city)
        return np.array(tour, dtype=np.int32)

    # initial tour
    tour = random_insertion_tour()
    report_best_tour(tour.copy())
    best_tour = tour.copy()
    best_len = tour_length(best_tour)
    eps = 1e-12

    while budget > 0:
        improved = False
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
            # double-bridge perturbation
            budget -= 5
            if budget <= 0:
                break
            if n >= 4:
                # choose 4 distinct random cut points
                inds = sorted(np.random.choice(n, 4, replace=False))
                a, b, c, d = inds
                # segments: [0:a], [a:b], [b:c], [c:d], [d:]
                new_tour = np.concatenate([tour[:a], tour[c:d], tour[b:c], tour[a:b], tour[d:]])
                if len(new_tour) == n:
                    tour = new_tour
                    new_len = tour_length(tour)
                    if new_len < best_len - eps:
                        best_len = new_len
                        best_tour = tour.copy()
                        report_best_tour(best_tour.copy())
                else:
                    # fallback: random restart
                    budget -= 1
                    tour = random_insertion_tour()
                    new_len = tour_length(tour)
                    if new_len < best_len - eps:
                        best_len = new_len
                        best_tour = tour.copy()
                        report_best_tour(best_tour.copy())
            else:
                # small n: random restart
                budget -= 1
                tour = random_insertion_tour()
                new_len = tour_length(tour)
                if new_len < best_len - eps:
                    best_len = new_len
                    best_tour = tour.copy()
                    report_best_tour(best_tour.copy())
        elif not improved:
            break
    return best_tour
import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        rng = np.random.default_rng(seed)
        tour = np.arange(n, dtype=np.int64)
        rng.shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)
    dist = distance_matrix

    def tour_length(t):
        return sum(dist[t[i], t[(i + 1) % n]] for i in range(n))

    def nearest_neighbor(start):
        tour = [start]
        visited = {start}
        cur = start
        while len(tour) < n:
            best = None
            best_d = np.inf
            for i in range(n):
                if i not in visited:
                    d = dist[cur, i]
                    if d < best_d:
                        best_d = d
                        best = i
            tour.append(best)
            visited.add(best)
            cur = best
        return np.array(tour, dtype=np.int64)

    start = rng.integers(n)
    tour = nearest_neighbor(start)
    best_tour = tour.copy()
    best_dist = tour_length(tour)
    cur_dist = best_dist
    report_best_tour(best_tour)
    ops = 0
    improved = True

    while ops < budget:
        if not improved:
            # Perturb: random 2-opt move
            i = rng.integers(0, n - 1)
            j = rng.integers(i + 2, n)
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            cur_dist = tour_length(tour)
            ops += 1
            improved = True
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            continue

        # First-improvement 2-opt pass
        improved = False
        for i in range(n - 1):
            if ops >= budget:
                break
            for j in range(i + 2, n):
                if ops >= budget:
                    break
                ops += 1
                a = tour[i]
                b = tour[(i + 1) % n]
                c = tour[j]
                d = tour[(j + 1) % n]
                old = dist[a, b] + dist[c, d]
                new_dist = dist[a, c] + dist[b, d]
                if new_dist + 1e-12 < old:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    cur_dist += new_dist - old
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour
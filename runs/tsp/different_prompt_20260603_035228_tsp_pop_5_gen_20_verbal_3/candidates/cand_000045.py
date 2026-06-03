import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        d = distance_matrix[t[-1], t[0]]
        for i in range(n - 1):
            d += distance_matrix[t[i], t[i + 1]]
        return d

    def two_opt_first(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[0]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[0]])
                    else:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[j+1]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[j+1]])
                    if delta > 1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return tour

    def perturbation(tour):
        seg_len = min(np.random.randint(1, n // 4 + 1), n - 1)
        i = np.random.randint(0, n - seg_len)
        tour[i:i+seg_len] = tour[i:i+seg_len][::-1]
        return tour

    # Initial tour: nearest neighbor from random start
    start = np.random.randint(n)
    tour = [start]
    visited = [False] * n
    visited[start] = True
    for _ in range(n - 1):
        last = tour[-1]
        best_next = -1
        bestd = float('inf')
        for j in range(n):
            if not visited[j] and distance_matrix[last, j] < bestd:
                bestd = distance_matrix[last, j]
                best_next = j
        tour.append(best_next)
        visited[best_next] = True
    tour = np.array(tour, dtype=int)
    best_dist = total_dist(tour)
    best_tour = tour.copy()
    report_best_tour(best_tour)

    # ILS loop without restarts
    max_iter = 30
    for _ in range(max_iter):
        tour = two_opt_first(tour)
        cur = total_dist(tour)
        if cur < best_dist:
            best_dist = cur
            best_tour = tour.copy()
            report_best_tour(best_tour)
        tour = perturbation(tour)
    return best_tour
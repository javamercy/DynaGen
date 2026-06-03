import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    if n == 2:
        tour = np.array([0, 1])
        report_best_tour(tour)
        return tour

    # Nearest neighbor construction
    tour = [0]
    visited = [False] * n
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        best = -1
        bestd = float('inf')
        for j in range(n):
            if not visited[j] and distance_matrix[last, j] < bestd:
                bestd = distance_matrix[last, j]
                best = j
        tour.append(best)
        visited[best] = True
    tour = np.array(tour, dtype=int)

    def total_dist(t):
        return distance_matrix[t[-1], t[0]] + np.sum(distance_matrix[t[:-1], t[1:]])

    best_tour = tour.copy()
    best_dist = total_dist(tour)
    report_best_tour(best_tour)

    max_cycles = 50
    for _ in range(max_cycles):
        # 2-opt first improvement
        improved = True
        while improved:
            improved = False
            cur_dist = total_dist(tour)
            for i in range(n - 2):
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
                        cur_dist -= delta
                        if cur_dist < best_dist - 1e-12:
                            best_dist = cur_dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                        break
                if improved:
                    break
        # Double-bridge perturbation
        if _ < max_cycles - 1:
            cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
            segments = [tour[:cuts[0]].tolist(), tour[cuts[0]:cuts[1]].tolist(),
                        tour[cuts[1]:cuts[2]].tolist(), tour[cuts[2]:].tolist()]
            new_tour = np.array(segments[0] + segments[2] + segments[1] + segments[3])
            tour = new_tour
            new_dist = total_dist(tour)
            if new_dist < best_dist - 1e-12:
                best_dist = new_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
    return best_tour
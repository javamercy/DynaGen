import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        tour = np.array([0], dtype=int)
        report_best_tour(tour.copy())
        return tour
    if n == 2:
        tour = np.array([0, 1], dtype=int)
        report_best_tour(tour.copy())
        return tour

    def total(t):
        return distance_matrix[t[-1], t[0]] + np.sum(distance_matrix[t[:-1], t[1:]])

    tour = list(np.random.permutation(n))
    best_tour = tour[:]
    best_dist = total(tour)
    report_best_tour(np.array(tour))

    improved = True
    while improved:
        improved = False
        for i in range(n):
            node = tour[i]
            new_tour = tour[:i] + tour[i+1:]
            for j in range(n):
                candidate = new_tour[:j] + [node] + new_tour[j:]
                cand_dist = total(candidate)
                if cand_dist < best_dist - 1e-8:
                    best_dist = cand_dist
                    best_tour = candidate[:]
                    tour = candidate[:]
                    improved = True
                    report_best_tour(np.array(best_tour))
                    break
            if improved:
                break
    return np.array(best_tour)
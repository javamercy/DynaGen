import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    # Random initial tour
    tour = np.random.permutation(n).astype(np.int32)
    report_best_tour(tour)
    # 2-opt improvement
    ext = np.empty(n+1, dtype=np.int32)
    ext[:n] = tour
    ext[n] = tour[0]
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                delta = (distance_matrix[ext[i], ext[j]] +
                         distance_matrix[ext[i+1], ext[j+1]] -
                         distance_matrix[ext[i], ext[i+1]] -
                         distance_matrix[ext[j], ext[j+1]])
                if delta < 0:
                    ext[i+1:j+1] = ext[i+1:j+1][::-1]
                    improved = True
                    new_tour = ext[:n].copy()
                    report_best_tour(new_tour)
                    break
            if improved:
                break
    return ext[:n]
import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    # Nearest neighbor construction
    tour = np.empty(n, dtype=np.int32)
    unvisited = np.ones(n, dtype=bool)
    tour[0] = 0
    unvisited[0] = False
    current = 0
    for i in range(1, n):
        dists = np.where(unvisited, distance_matrix[current], np.inf)
        next_node = np.argmin(dists)
        tour[i] = next_node
        unvisited[next_node] = False
        current = next_node
    report_best_tour(tour)
    # 2-opt with full scan per pass, max passes = 2*n
    ext = np.empty(n+1, dtype=np.int32)
    ext[:n] = tour
    ext[n] = tour[0]
    max_passes = 2 * n
    for _ in range(max_passes):
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
        if not improved:
            break
    return ext[:n]
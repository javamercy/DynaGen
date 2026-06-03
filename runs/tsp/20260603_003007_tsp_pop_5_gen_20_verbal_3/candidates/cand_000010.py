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
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                delta = (distance_matrix[tour[i], tour[j]] +
                         distance_matrix[tour[(i+1)%n], tour[(j+1)%n]] -
                         distance_matrix[tour[i], tour[(i+1)%n]] -
                         distance_matrix[tour[j], tour[(j+1)%n]])
                if delta < 0:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    report_best_tour(tour.copy())
                    break
            if improved:
                break
    return tour
import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    tour = [0, 1]
    unvisited = set(range(2, n))
    while unvisited:
        best = (np.inf, None, None)
        for node in unvisited:
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                delta = distance_matrix[tour[i], node] + distance_matrix[node, tour[j]] - distance_matrix[tour[i], tour[j]]
                if delta < best[0]:
                    best = (delta, node, i + 1)
        tour.insert(best[2], best[1])
        unvisited.remove(best[1])
    best_tour = np.array(tour)
    report_best_tour(best_tour)
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 2, n):
                a, b, c, d = tour[i], tour[(i + 1) % n], tour[k], tour[(k + 1) % n]
                if distance_matrix[a, b] + distance_matrix[c, d] > distance_matrix[a, c] + distance_matrix[b, d]:
                    tour[i + 1:k + 1] = reversed(tour[i + 1:k + 1])
                    improved = True
        if improved:
            best_tour = np.array(tour)
            report_best_tour(best_tour)
    return best_tour
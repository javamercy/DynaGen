import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    # nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        dists = distance_matrix[current, list(unvisited)]
        nearest = list(unvisited)[np.argmin(dists)]
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
        report_best_tour(np.array(tour, dtype=np.int32))
    tour = np.array(tour, dtype=np.int32)
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
                if delta < -1e-12:
                    ext[i+1:j+1] = ext[i+1:j+1][::-1]
                    improved = True
                    report_best_tour(ext[:n].copy())
                    break
            if improved:
                break
    return ext[:n]
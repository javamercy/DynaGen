import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    # farthest insertion construction
    tour = [0]
    unvisited = np.ones(n, dtype=bool)
    unvisited[0] = False
    while unvisited.any():
        # compute min distance from each unvisited node to current tour
        min_dists = np.min(distance_matrix[unvisited][:, tour], axis=1)
        farthest_node = np.where(unvisited)[0][np.argmax(min_dists)]
        # find best insertion position
        best_delta = np.inf
        best_pos = 0
        for i in range(len(tour)):
            prev = tour[i]
            next_city = tour[(i+1) % len(tour)]
            delta = distance_matrix[prev, farthest_node] + distance_matrix[farthest_node, next_city] - distance_matrix[prev, next_city]
            if delta < best_delta:
                best_delta = delta
                best_pos = i+1
        tour.insert(best_pos, farthest_node)
        unvisited[farthest_node] = False
        report_best_tour(np.array(tour, dtype=np.int32))
    tour = np.array(tour, dtype=np.int32)
    # 2-opt on extended tour
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
import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    best_dist = np.inf
    best_tour = None
    num_starts = min(n, 8)
    for start in range(num_starts):
        # Nearest neighbor construction from start
        tour = np.empty(n, dtype=np.int32)
        unvisited = np.ones(n, dtype=bool)
        tour[0] = start
        unvisited[start] = False
        current = start
        for i in range(1, n):
            dists = np.where(unvisited, distance_matrix[current], np.inf)
            next_node = np.argmin(dists)
            tour[i] = next_node
            unvisited[next_node] = False
            current = next_node
        dist = distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # 2-opt improvement
        ext = np.empty(n+1, dtype=np.int32)
        ext[:n] = tour
        ext[n] = tour[0]
        max_passes = n
        for _ in range(max_passes):
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    delta = (distance_matrix[ext[i], ext[j]] +
                             distance_matrix[ext[i+1], ext[j+1]] -
                             distance_matrix[ext[i], ext[i+1]] -
                             distance_matrix[ext[j], ext[j+1]])
                    if delta < -1e-10:
                        ext[i+1:j+1] = ext[i+1:j+1][::-1]
                        improved = True
                        new_tour = ext[:n].copy()
                        new_dist = distance_matrix[new_tour[-1], new_tour[0]] + np.sum(distance_matrix[new_tour[:-1], new_tour[1:]])
                        if new_dist < best_dist - 1e-10:
                            best_dist = new_dist
                            best_tour = new_tour.copy()
                            report_best_tour(best_tour)
            if not improved:
                break
    return best_tour
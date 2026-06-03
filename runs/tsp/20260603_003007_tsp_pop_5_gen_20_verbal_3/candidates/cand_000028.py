import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = np.inf
    for start in range(n):
        unvisited = np.ones(n, dtype=bool)
        unvisited[start] = False
        tour = [start]
        current = start
        while unvisited.any():
            dists = distance_matrix[current]
            dists[~unvisited] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            unvisited[next_node] = False
            current = next_node
        tour_arr = np.array(tour, dtype=np.int32)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour_arr[i], tour_arr[(i+1)%n]]
        if dist < best_dist:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
    # 2-opt improvement
    n = len(best_tour)
    ext = np.empty(n+1, dtype=np.int32)
    ext[:n] = best_tour
    ext[n] = best_tour[0]
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
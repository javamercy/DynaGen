import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    # Cheapest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    # start with 0 and its nearest neighbor
    nearest = min(range(1, n), key=lambda x: distance_matrix[0, x])
    tour.append(nearest)
    unvisited.remove(nearest)
    while unvisited:
        best_city = None
        best_increase = np.inf
        best_pos = -1
        for city in unvisited:
            for pos in range(len(tour)):
                i = tour[pos]
                j = tour[(pos + 1) % len(tour)]
                increase = distance_matrix[i, city] + distance_matrix[city, j] - distance_matrix[i, j]
                if increase < best_increase:
                    best_increase = increase
                    best_city = city
                    best_pos = pos + 1
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour, dtype=np.int32)
    report_best_tour(tour)
    # 2-opt improvement
    ext = np.empty(n + 1, dtype=np.int32)
    ext[:n] = tour
    ext[n] = tour[0]
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 2, n):
                delta = (distance_matrix[ext[i], ext[j]] +
                         distance_matrix[ext[i + 1], ext[j + 1]] -
                         distance_matrix[ext[i], ext[i + 1]] -
                         distance_matrix[ext[j], ext[j + 1]])
                if delta < 0:
                    ext[i + 1:j + 1] = ext[i + 1:j + 1][::-1]
                    improved = True
                    new_tour = ext[:n].copy()
                    report_best_tour(new_tour)
                    break
            if improved:
                break
    return ext[:n]
import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    if n == 2:
        tour = np.array([0, 1])
        report_best_tour(tour)
        return tour

    # Find farthest pair
    max_dist = -1
    start_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i, j] > max_dist:
                max_dist = distance_matrix[i, j]
                start_pair = (i, j)
    tour = [start_pair[0], start_pair[1]]
    unvisited = set(range(n)) - set(tour)

    while unvisited:
        best_city = None
        best_pos = None
        best_increase = float('inf')
        m = len(tour)
        for city in unvisited:
            for i in range(m):
                prev = tour[i]
                nxt = tour[(i + 1) % m]
                increase = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                if increase < best_increase:
                    best_increase = increase
                    best_city = city
                    best_pos = i + 1
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)

    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    return tour_arr
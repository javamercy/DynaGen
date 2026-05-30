import numpy as np
def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    start = 0
    end = np.argmax(distance_matrix[start])
    tour = [start, end]
    in_tour = {start, end}
    while len(tour) < n:
        best_candidate = -1
        best_increase = float('inf')
        best_pos = -1
        for node in range(n):
            if node in in_tour:
                continue
            min_increase = float('inf')
            min_pos = -1
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i+1)%len(tour)]
                inc = distance_matrix[a][node] + distance_matrix[node][b] - distance_matrix[a][b]
                if inc < min_increase:
                    min_increase = inc
                    min_pos = i+1
            if min_increase < best_increase:
                best_increase = min_increase
                best_candidate = node
                best_pos = min_pos
        tour.insert(best_pos, best_candidate)
        in_tour.add(best_candidate)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    return tour_arr
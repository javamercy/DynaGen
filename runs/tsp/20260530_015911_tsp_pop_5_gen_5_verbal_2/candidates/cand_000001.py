import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])
    # Construction: farthest insertion
    max_dist = -1
    start_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i][j] > max_dist:
                max_dist = distance_matrix[i][j]
                start_pair = (i, j)
    tour = [start_pair[0], start_pair[1]]
    unvisited = set(range(n)) - set(tour)
    while unvisited:
        best_node = None
        best_increase = -1
        best_pos = None
        for node in unvisited:
            min_increase = float('inf')
            min_pos = None
            m = len(tour)
            for i in range(m):
                j = (i + 1) % m
                inc = distance_matrix[tour[i]][node] + distance_matrix[node][tour[j]] - distance_matrix[tour[i]][tour[j]]
                if inc < min_increase:
                    min_increase = inc
                    min_pos = j
            if min_increase > best_increase:
                best_increase = min_increase
                best_node = node
                best_pos = min_pos
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
    tour = np.array(tour)
    best_dist = _tour_length(distance_matrix, tour)
    report_best_tour(tour.copy())
    # Improvement: 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n-1):
                a = tour[i]
                b = tour[i+1]
                c = tour[j]
                d = tour[(j+1) % n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    new_dist = _tour_length(distance_matrix, tour)
                    if new_dist < best_dist - 1e-12:
                        best_dist = new_dist
                        report_best_tour(tour.copy())
                    break
            if improved:
                break
    return tour

def _tour_length(dm, tour):
    n = len(tour)
    total = 0.0
    for k in range(n):
        total += dm[tour[k]][tour[(k+1) % n]]
    return total
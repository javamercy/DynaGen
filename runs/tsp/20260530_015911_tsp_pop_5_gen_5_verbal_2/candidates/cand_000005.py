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
    def _tour_length(t):
        total = 0.0
        for k in range(n):
            total += distance_matrix[t[k]][t[(k+1) % n]]
        return total
    best_dist = _tour_length(tour)
    report_best_tour(tour.copy())
    # VND: 2-opt and Or-opt
    improved = True
    while improved:
        improved = False
        # 2-opt
        inner_improved = True
        while inner_improved:
            inner_improved = False
            for i in range(n-2):
                for j in range(i+2, n-1):
                    a = tour[i]
                    b = tour[i+1]
                    c = tour[j]
                    d = tour[(j+1) % n]
                    delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        inner_improved = True
                        new_dist = _tour_length(tour)
                        if new_dist < best_dist - 1e-12:
                            best_dist = new_dist
                            report_best_tour(tour.copy())
                        break
                if inner_improved:
                    break
        # Or-opt
        or_improved = False
        for i in range(n):
            node = tour[i]
            tour_no_i = np.delete(tour, i)
            for j in range(n):  # insert at any position
                candidate = np.insert(tour_no_i, j, node)
                cand_dist = _tour_length(candidate)
                if cand_dist < best_dist - 1e-12:
                    tour = candidate
                    best_dist = cand_dist
                    report_best_tour(tour.copy())
                    or_improved = True
                    break
            if or_improved:
                break
        if or_improved:
            improved = True
    return tour
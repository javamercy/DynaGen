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
        best_node = -1
        best_dist = -1
        best_pos = 0
        for node in range(n):
            if node in in_tour:
                continue
            min_dist = float('inf')
            for t in tour:
                d = distance_matrix[node][t]
                if d < min_dist:
                    min_dist = d
            if min_dist > best_dist:
                best_dist = min_dist
                best_node = node
                # find insertion position that minimizes cost increase
        # after selecting best_node, find best insertion position
        min_increase = float('inf')
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i+1)%len(tour)]
            inc = distance_matrix[a][best_node] + distance_matrix[best_node][b] - distance_matrix[a][b]
            if inc < min_increase:
                min_increase = inc
                best_pos = i+1
        tour.insert(best_pos, best_node)
        in_tour.add(best_node)
    tour = np.array(tour)
    report_best_tour(tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                if j == i+1:
                    continue
                # compute potential improvement
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-12:
                    # reverse segment (i+1)..j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    report_best_tour(tour)
    return tour
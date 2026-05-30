import numpy as np
def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # farthest insertion construction
    start = 0
    end = np.argmax(distance_matrix[start])
    tour = [start, end]
    in_tour = {start, end}
    while len(tour) < n:
        best_node = -1
        best_dist = -1.0
        for node in range(n):
            if node in in_tour:
                continue
            min_dist = min(distance_matrix[node][t] for t in tour)
            if min_dist > best_dist:
                best_dist = min_dist
                best_node = node
        best_pos = -1
        best_increase = float('inf')
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i+1)%len(tour)]
            increase = distance_matrix[a][best_node] + distance_matrix[best_node][b] - distance_matrix[a][b]
            if increase < best_increase:
                best_increase = increase
                best_pos = i+1
        tour.insert(best_pos, best_node)
        in_tour.add(best_node)
    tour = np.array(tour)
    report_best_tour(tour)
    # limited 2-opt improvement
    max_passes = 10
    for _ in range(max_passes):
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < 0:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    report_best_tour(tour)
                    break
            if improved:
                break
        if not improved:
            break
    return tour
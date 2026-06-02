import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour.copy())
        return tour
    # farthest insertion construction
    # find farthest pair
    max_dist = 0
    start = end = 0
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i, j]
            if d > max_dist:
                max_dist = d
                start, end = i, j
    tour = [start, end]
    remaining = set(range(n)) - {start, end}
    while remaining:
        # farthest remaining node from current tour
        farthest_dist = -1
        farthest_node = None
        for node in remaining:
            min_dist = min(distance_matrix[node, t] for t in tour)
            if min_dist > farthest_dist:
                farthest_dist = min_dist
                farthest_node = node
        # best insertion position
        best_increase = float('inf')
        best_pos = 0
        for i in range(len(tour)):
            prev = tour[i]
            nxt = tour[(i+1) % len(tour)]
            increase = distance_matrix[prev, farthest_node] + distance_matrix[farthest_node, nxt] - distance_matrix[prev, nxt]
            if increase < best_increase:
                best_increase = increase
                best_pos = i+1
        tour.insert(best_pos, farthest_node)
        remaining.remove(farthest_node)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr.copy())
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                if j - i == 1:
                    continue
                i1 = i
                i2 = i+1
                j1 = j
                j2 = (j+1) % n
                if distance_matrix[tour[i1], tour[i2]] + distance_matrix[tour[j1], tour[j2]] > distance_matrix[tour[i1], tour[j1]] + distance_matrix[tour[i2], tour[j2]]:
                    improved = True
                    tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                    tour_arr = np.array(tour)
                    report_best_tour(tour_arr.copy())
    return np.array(tour)
import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # Farthest insertion construction
    max_dist = -1
    start = 0
    end = 1
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i, j] > max_dist:
                max_dist = distance_matrix[i, j]
                start, end = i, j
    tour = [start, end]
    in_tour = {start, end}
    while len(tour) < n:
        farthest_city = None
        max_min_dist = -1
        for city in range(n):
            if city in in_tour:
                continue
            min_dist = min(distance_matrix[city, t] for t in tour)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                farthest_city = city
        best_pos = 0
        best_increase = float('inf')
        for pos in range(len(tour)):
            prev = tour[pos]
            nxt = tour[(pos+1) % len(tour)]
            increase = distance_matrix[prev, farthest_city] + distance_matrix[farthest_city, nxt] - distance_matrix[prev, nxt]
            if increase < best_increase:
                best_increase = increase
                best_pos = pos+1
        tour.insert(best_pos, farthest_city)
        in_tour.add(farthest_city)
    best_tour = np.array(tour)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    # Check if this new tour is better
                    new_dist = distance_matrix[tour[i], tour[i+1]] + distance_matrix[tour[j], tour[(j+1)%n]]
                    cur_dist = sum(distance_matrix[tour[k], tour[(k+1)%n]] for k in range(n))
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = np.array(tour)
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour
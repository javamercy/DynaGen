import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)
    # Initialize tour with two farthest cities
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
    # Farthest insertion construction
    while len(tour) < n:
        # Find farthest city not in tour
        farthest_city = None
        max_min_dist = -1
        for city in range(n):
            if city in in_tour:
                continue
            # min distance to any city in tour
            min_dist = min(distance_matrix[city, tour[i]] for i in range(len(tour)))
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                farthest_city = city
        # Insert farthest_city at best position
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
    # Simple 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    break
            if improved:
                break
    report_best_tour(np.array(tour))
    return np.array(tour)
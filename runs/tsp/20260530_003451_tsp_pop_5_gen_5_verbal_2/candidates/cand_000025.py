import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)
    
    # Farthest insertion construction
    max_dist = -1
    start, end = 0, 1
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
    
    best_tour = tour.copy()
    best_dist = tour_distance(tour, distance_matrix)
    report_best_tour(np.array(best_tour))
    
    # Iterative 2-opt with perturbation escapes
    max_iterations = 100
    for _ in range(max_iterations):
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
        cur_dist = tour_distance(tour, distance_matrix)
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(np.array(best_tour))
        # Double-bridge perturbation
        if n >= 8:
            # choose two random split points
            p = sorted(random.sample(range(1, n-1), 3))
            i1, i2, i3 = p[0], p[1], p[2]
            new_tour = tour[:i1] + tour[i3:] + tour[i2:i3] + tour[i1:i2]
            tour = new_tour
        else:
            # shuffle two random segments for small n
            i = random.randint(1, n//2)
            j = random.randint(i+1, n-1)
            tour[i:j+1] = reversed(tour[i:j+1])
    
    return np.array(best_tour)

def tour_distance(tour, dist):
    return sum(dist[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))
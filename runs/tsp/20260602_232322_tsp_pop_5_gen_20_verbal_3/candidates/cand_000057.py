import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # Multi-start nearest neighbor construction
    best_tour = None
    best_dist = float('inf')
    num_starts = min(5, n)
    for start in random.sample(range(n), num_starts):
        tour_list = [start]
        visited = {start}
        current = start
        while len(tour_list) < n:
            next_city = min((j for j in range(n) if j not in visited), key=lambda j: distance_matrix[current, j])
            tour_list.append(next_city)
            visited.add(next_city)
            current = next_city
        tour = np.array(tour_list)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i + 1) % n]]
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    tour = best_tour.copy()
    current_dist = best_dist
    # Simulated annealing
    T = distance_matrix.max() * n / 10.0
    cooling = 0.999
    for _ in range(n * 100):
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
        new_dist = 0.0
        for k in range(n):
            new_dist += distance_matrix[tour[k], tour[(k + 1) % n]]
        if new_dist < current_dist or random.random() < np.exp((current_dist - new_dist) / T):
            current_dist = new_dist
            if current_dist < best_dist - 1e-10:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
        else:
            tour[i], tour[j] = tour[j], tour[i]  # revert
        T *= cooling
    return best_tour
import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # Nearest neighbor construction from random start
    start = random.randrange(n)
    tour = [start]
    remaining = set(range(n))
    remaining.remove(start)
    current = start
    while remaining:
        next_city = min(remaining, key=lambda c: distance_matrix[current, c])
        tour.append(next_city)
        remaining.remove(next_city)
        current = next_city
    tour = np.array(tour)
    dist = 0.0
    for i in range(n):
        dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+1, n):
                if j - i == 1 or (i == 0 and j == n-1):
                    continue
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                if delta < -1e-10:
                    # reverse segment i+1 to j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    dist += delta
                    improved = True
                    if dist < best_dist - 1e-10:
                        best_dist = dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
        if not improved:
            break
    return tour
import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = np.inf
    for _ in range(5):
        start = random.randint(0, n-1)
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        current = start
        while unvisited:
            next_city = min(unvisited, key=lambda c: distance_matrix[current, c])
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        tour = np.array(tour, dtype=np.int64)
        dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = np.flip(tour[i+1:j+1])
                        dist += delta
                        if dist < best_dist - 1e-10:
                            best_dist = dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
    return best_tour
import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour.copy())
        return tour
    # Farthest insertion
    i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
    if i > j:
        i, j = j, i
    tour = [i, j]
    unvisited = set(range(n)) - {i, j}
    while unvisited:
        max_dist = -1
        farthest = None
        for city in unvisited:
            dist = min(distance_matrix[city, t] for t in tour)
            if dist > max_dist:
                max_dist = dist
                farthest = city
        best_cost = np.inf
        best_pos = 0
        for pos in range(len(tour) + 1):
            if pos == 0:
                a = tour[-1]
                b = tour[0]
            elif pos == len(tour):
                a = tour[-1]
                b = tour[0]
            else:
                a = tour[pos-1]
                b = tour[pos]
            cost = distance_matrix[a, farthest] + distance_matrix[farthest, b] - distance_matrix[a, b]
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        tour.insert(best_pos, farthest)
        unvisited.remove(farthest)
    tour = np.array(tour, dtype=int)
    best_tour = tour.copy()
    best_dist = distance_matrix[tour[:-1], tour[1:]].sum() + distance_matrix[tour[-1], tour[0]]
    report_best_tour(tour.copy())
    
    def two_opt(tour, current_dist):
        nonlocal best_tour, best_dist
        improved = True
        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    a, b = tour[i], tour[(i+1) % n]
                    c, d = tour[j], tour[(j+1) % n]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]
                             - distance_matrix[a, b] - distance_matrix[c, d])
                    if delta < -1e-12:
                        tour[i+1:j+1] = np.flip(tour[i+1:j+1])
                        current_dist += delta
                        if current_dist < best_dist:
                            best_dist = current_dist
                            best_tour = tour.copy()
                            report_best_tour(tour.copy())
                        improved = True
                        break
                if improved:
                    break
        return current_dist
    
    current_dist = best_dist
    current_dist = two_opt(tour, current_dist)
    if n > 3:
        i = random.randint(0, n-2)
        j = random.randint(i+2, n-1)
        tour[i+1:j+1] = np.flip(tour[i+1:j+1])
        current_dist = distance_matrix[tour[:-1], tour[1:]].sum() + distance_matrix[tour[-1], tour[0]]
        if current_dist < best_dist:
            best_dist = current_dist
            best_tour = tour.copy()
            report_best_tour(tour.copy())
        two_opt(tour, current_dist)
    return best_tour
import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    
    def tour_dist(t):
        d = 0.0
        for i in range(n):
            d += distance_matrix[t[i]][t[(i+1)%n]]
        return d
    
    best_tour = None
    best_dist = float('inf')
    
    def two_opt(tour):
        nonlocal best_tour, best_dist
        L = len(tour)
        improved = True
        passes = 0
        while improved and passes < 20:
            improved = False
            passes += 1
            order_i = list(range(L-1))
            random.shuffle(order_i)
            for i in order_i:
                for j in range(i+2, L):
                    a = tour[i]
                    b = tour[(i+1)%L]
                    c = tour[j%L]
                    d = tour[(j+1)%L]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old - 1e-12:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        new_dist = tour_dist(tour)
                        if new_dist < best_dist - 1e-12:
                            best_dist = new_dist
                            best_tour = tour[:]
                            report_best_tour(np.array(tour))
                        break
                if improved:
                    break
        return tour
    
    def double_bridge(tour):
        L = len(tour)
        i = random.randint(1, L-4)
        j = random.randint(i+1, L-3)
        k = random.randint(j+1, L-2)
        new_tour = tour[:i] + tour[k:] + tour[j:k] + tour[i:j]
        return new_tour
    
    num_restarts = max(1, min(5, n // 100))
    for restart in range(num_restarts):
        tour = list(range(n))
        random.shuffle(tour)
        # report initial tour as potential best
        d = tour_dist(tour)
        if d < best_dist:
            best_dist = d
            best_tour = tour[:]
            report_best_tour(np.array(tour))
        tour = two_opt(tour)
        # perturbations
        for _ in range(5):
            # random swap perturbation
            if random.random() < 0.5:
                # swap two random cities
                a, b = random.sample(range(n), 2)
                tour[a], tour[b] = tour[b], tour[a]
            else:
                tour = double_bridge(tour)
            tour = two_opt(tour)
            d = tour_dist(tour)
            if d < best_dist:
                best_dist = d
                best_tour = tour[:]
                report_best_tour(np.array(tour))
    return np.array(best_tour)
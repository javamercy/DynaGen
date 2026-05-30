import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    # Nearest neighbor construction from random start
    start = random.randrange(n)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda x: distance_matrix[current][x])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    best_tour = tour[:]
    best_dist = tour_distance(tour, distance_matrix)
    report_best_tour(np.array(best_tour))
    
    def tour_distance(t, dm):
        d = 0.0
        for i in range(n):
            d += dm[t[i]][t[(i+1)%n]]
        return d
    
    def two_opt_delta(t, dm):
        nonlocal best_tour, best_dist
        improved = True
        while improved:
            improved = False
            L = len(t)
            for i in range(L-1):
                a = t[i]
                b = t[(i+1)%L]
                for j in range(i+2, L):
                    c = t[j%L]
                    d = t[(j+1)%L]
                    old = dm[a][b] + dm[c][d]
                    new = dm[a][c] + dm[b][d]
                    if new < old - 1e-12:
                        t[i+1:j+1] = reversed(t[i+1:j+1])
                        improved = True
                        new_dist = best_dist - old + new
                        if new_dist < best_dist - 1e-12:
                            best_dist = new_dist
                            best_tour = t[:]
                            report_best_tour(np.array(t))
                        # break out of loops to restart scanning
                        j = L
                        i = L-1
                        break
        return t
    
    tour = two_opt_delta(tour, distance_matrix)
    # Intensification: few random 2-opt perturbations
    for _ in range(5):
        L = len(tour)
        i = random.randint(0, L-3)
        j = random.randint(i+2, L-1)
        # perform a random 2-opt move (reverse segment)
        new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
        new_tour = two_opt_delta(new_tour, distance_matrix)
        new_dist = tour_distance(new_tour, distance_matrix)
        if new_dist < best_dist - 1e-12:
            best_dist = new_dist
            best_tour = new_tour[:]
            report_best_tour(np.array(new_tour))
        tour = new_tour
    return np.array(best_tour)
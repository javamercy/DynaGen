import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        return tour
    # Randomized greedy construction
    unvisited = set(range(n))
    start = random.randrange(n)
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        current = tour[-1]
        # Get distances to unvisited
        cand = list(unvisited)
        dists = [(distance_matrix[current, c], c) for c in cand]
        dists.sort()
        # Choose among the 3 nearest (or less if fewer)
        k = min(3, len(dists))
        choice = random.randrange(k)
        next_city = dists[choice][1]
        tour.append(next_city)
        unvisited.remove(next_city)
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    best_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    # 2-opt improvement
    iteration = 0
    improved = True
    while improved and iteration < budget:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                iteration += 1
                if iteration >= budget:
                    break
                # Check if reversing segment (i+1..j) reduces distance
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-12:
                    # Apply 2-opt move
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    best_tour = tour.copy()
                    # Update best_dist (not necessary but for potential use)
                    best_dist = best_dist + delta
                    improved = True
                    report_best_tour(best_tour)
                    break
            if improved or iteration >= budget:
                break
    return best_tour
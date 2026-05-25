import numpy as np
import random
import math

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=int)
        try:
            report_best_tour(tour)
        except:
            pass
        return tour
    rng = random.Random(seed)
    # Nearest neighbor construction
    start = rng.randrange(n)
    tour = [start]
    visited = {start}
    for _ in range(n-1):
        last = tour[-1]
        best = None
        best_dist = float('inf')
        for j in range(n):
            if j not in visited:
                d = distance_matrix[last, j]
                if d < best_dist:
                    best_dist = d
                    best = j
        tour.append(best)
        visited.add(best)
    tour = np.array(tour, dtype=int)
    # Compute initial distance
    current_dist = 0.0
    for i in range(n):
        current_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    best_tour = tour.copy()
    best_dist = current_dist
    try:
        report_best_tour(best_tour)
    except:
        pass
    # Simulated annealing
    evals = 0
    max_evals = budget
    initial_temp = max(1.0, n * 5.0)
    while evals < max_evals:
        # Generate random 2-opt move
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i > j:
            i, j = j, i
        if (j - i) < 2 or (i == 0 and j == n-1):
            continue
        a = tour[i]
        b = tour[(i+1)%n]
        c = tour[j]
        d = tour[(j+1)%n]
        old = distance_matrix[a, b] + distance_matrix[c, d]
        new = distance_matrix[a, c] + distance_matrix[b, d]
        delta = new - old
        evals += 1
        temp = initial_temp * (1 - evals / max_evals)
        if delta < -1e-12 or (temp > 0 and rng.random() < math.exp(-delta / temp)):
            # Accept move
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            current_dist += delta
            if current_dist < best_dist - 1e-12:
                best_dist = current_dist
                best_tour = tour.copy()
                try:
                    report_best_tour(best_tour)
                except:
                    pass
    return best_tour
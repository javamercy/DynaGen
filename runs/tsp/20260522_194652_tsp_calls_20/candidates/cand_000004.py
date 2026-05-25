import numpy as np
import random
import math

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.array([0, 1] if n == 2 else [0], dtype=int)
        if 'report_best_tour' in globals():
            globals()['report_best_tour'](tour)
        return tour
    random.seed(seed)
    dm = distance_matrix
    # Greedy nearest neighbor from random start
    start = random.randrange(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    current = start
    while unvisited:
        best_dist = float('inf')
        best_city = -1
        for city in unvisited:
            d = dm[current, city]
            if d < best_dist:
                best_dist = d
                best_city = city
        tour.append(best_city)
        unvisited.remove(best_city)
        current = best_city
    initial_tour = np.array(tour, dtype=int)
    if 'report_best_tour' in globals():
        globals()['report_best_tour'](initial_tour)
    # Simulated annealing
    best_tour = initial_tour.copy()
    best_len = _tour_length(best_tour, dm)
    current_tour = best_tour.copy()
    current_len = best_len
    T = best_len * 0.2
    T_min = 1e-6
    alpha = max(0.99, 1.0 - 1.0 / budget) if budget > 0 else 0.99
    for iteration in range(budget):
        if T < T_min:
            break
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i > j:
            i, j = j, i
        if i == j or (j - i) == 1 or (i == 0 and j == n-1):
            continue
        a = current_tour[i]
        b = current_tour[(i+1) % n]
        c = current_tour[j]
        d = current_tour[(j+1) % n]
        delta = dm[a][c] + dm[b][d] - dm[a][b] - dm[c][d]
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_tour = np.concatenate([
                current_tour[:i+1],
                current_tour[i+1:j+1][::-1],
                current_tour[j+1:]
            ])
            current_len += delta
            if current_len < best_len:
                best_len = current_len
                best_tour = current_tour.copy()
                if 'report_best_tour' in globals():
                    globals()['report_best_tour'](best_tour)
        T *= alpha
    return best_tour

def _tour_length(tour, dm):
    n = len(tour)
    length = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i+1) % n]
        length += dm[a][b]
    return length
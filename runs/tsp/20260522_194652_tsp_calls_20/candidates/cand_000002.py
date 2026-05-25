import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.array([0, 1] if n == 2 else [0])
        from builtins import globals as _globals
        if 'report_best_tour' in _globals():
            _globals()['report_best_tour'](tour)
        return tour
    random.seed(seed)
    dm = distance_matrix
    # Cheapest insertion on random permutation
    start = random.randrange(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    second = random.choice(list(unvisited))
    tour.append(second)
    unvisited.remove(second)
    remaining = list(unvisited)
    random.shuffle(remaining)
    for city in remaining:
        best_cost = float('inf')
        best_pos = 0
        L = len(tour)
        for i in range(L):
            a = tour[i]
            b = tour[(i+1) % L]
            cost = dm[a][city] + dm[city][b] - dm[a][b]
            if cost < best_cost:
                best_cost = cost
                best_pos = i+1
        tour.insert(best_pos, city)
    initial_tour = np.array(tour)
    def tour_length(t):
        length = 0.0
        for i in range(len(t)):
            a = t[i]
            b = t[(i+1) % len(t)]
            length += dm[a][b]
        return length
    best_tour = initial_tour.copy()
    best_len = tour_length(best_tour)
    if 'report_best_tour' in globals():
        globals()['report_best_tour'](best_tour)
    # Random 2-opt improvement
    tour_list = best_tour.tolist()
    n_tour = n
    max_attempts = min(10000, max(100, budget // 2))
    for _ in range(max_attempts):
        i = random.randint(0, n_tour - 1)
        j = random.randint(0, n_tour - 1)
        if i > j:
            i, j = j, i
        if i == j:
            continue
        if (j - i) == 1 or (i == 0 and j == n_tour - 1):
            continue
        a = tour_list[i]
        b = tour_list[(i+1) % n_tour]
        c = tour_list[j]
        d = tour_list[(j+1) % n_tour]
        delta = dm[a][c] + dm[b][d] - dm[a][b] - dm[c][d]
        if delta < -1e-12:
            # reverse segment
            tour_list[i+1 : j+1] = reversed(tour_list[i+1 : j+1])
            best_len += delta
            # update best if improvement is overall best (but tours are monotonic)
            if best_len < tour_length(tour_list):
                best_len = tour_length(tour_list)
            if 'report_best_tour' in globals():
                globals()['report_best_tour'](np.array(tour_list))
    return np.array(tour_list)
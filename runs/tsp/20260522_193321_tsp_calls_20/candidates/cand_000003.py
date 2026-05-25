import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=int)
        # report initial tour
        report_best_tour(tour)
        return tour

    rng = random.Random(seed)
    # initial random tour
    tour = list(range(n))
    rng.shuffle(tour)
    tour = np.array(tour, dtype=int)

    def tour_distance(t):
        d = 0.0
        for i in range(n-1):
            d += distance_matrix[t[i], t[i+1]]
        d += distance_matrix[t[-1], t[0]]
        return d

    best_tour = tour.copy()
    best_dist = tour_distance(tour)
    report_best_tour(tour)

    # local improvement: 2-opt
    def improve(t):
        nonlocal budget, best_tour, best_dist
        improved = True
        while improved and budget > 0:
            improved = False
            for i in range(n-2):
                if budget <= 0:
                    break
                for j in range(i+2, n-1):
                    if budget <= 0:
                        break
                    # reverse segment [i+1, j]
                    new_t = t.copy()
                    new_t[i+1:j+1] = t[i+1:j+1][::-1]
                    new_d = tour_distance(new_t)
                    budget -= 1
                    if new_d < best_dist:
                        best_tour = new_t.copy()
                        best_dist = new_d
                        t = new_t
                        improved = True
                        report_best_tour(best_tour)
                        break
                if improved:
                    break
        return t

    # perturbation: double-bridge
    def perturb(t):
        a = rng.randint(1, n//3)
        b = rng.randint(a+1, 2*n//3)
        c = rng.randint(b+1, n-1)
        seg1 = t[:a]
        seg2 = t[a:b]
        seg3 = t[b:c]
        seg4 = t[c:]
        return np.concatenate([seg1, seg3, seg2, seg4])

    # main ILS loop
    while budget > 0:
        tour = improve(tour)
        if budget <= 0:
            break
        tour = perturb(tour)
        # budget not consumed by perturbation itself

    return best_tour
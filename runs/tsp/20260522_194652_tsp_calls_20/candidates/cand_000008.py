import numpy as np
import random

def solve_tsp(distance_matrix, seed, budget):
    n = distance_matrix.shape[0]
    rng = random.Random(seed)

    if n <= 2:
        tour = list(range(n))
        report_best_tour(tour)
        return np.array(tour)

    def compute_tour_dist(t, mat):
        d = 0
        for i in range(n):
            d += mat[t[i], t[(i+1)%n]]
        return d

    # Nearest neighbor initial tour
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda c: distance_matrix[current, c])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    best_tour = tour[:]
    best_dist = compute_tour_dist(tour, distance_matrix)
    report_best_tour(tour)

    def two_opt(t, budget_left):
        improve = True
        while improve and budget_left > 0:
            improve = False
            for i in range(n-2):
                for j in range(i+2, n):
                    if budget_left <= 0:
                        break
                    budget_left -= 1
                    a = t[i]
                    b = t[i+1]
                    c = t[j]
                    d = t[(j+1)%n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        t[i+1:j+1] = reversed(t[i+1:j+1])
                        improve = True
                        break
                if improve:
                    break
        return t, budget_left

    # Main ILS loop
    current_tour = tour[:]
    current_dist = best_dist
    budget_remaining = budget

    while budget_remaining > 0:
        # Local search
        improved_tour, budget_remaining = two_opt(current_tour, budget_remaining)
        new_dist = compute_tour_dist(improved_tour, distance_matrix)
        if new_dist < best_dist - 1e-12:
            best_tour = improved_tour[:]
            best_dist = new_dist
            report_best_tour(best_tour)
        current_tour = improved_tour[:]
        current_dist = new_dist
        if budget_remaining <= 0:
            break

        # Perturbation: random segment reversal and a city swap
        # Reversal: choose two random indices and reverse segment
        i = rng.randint(0, n-2)
        j = rng.randint(i+1, n-1)
        current_tour[i:j+1] = reversed(current_tour[i:j+1])
        # Swap: choose two random indices (may be same as previous, but that's okay)
        a = rng.randint(0, n-1)
        b = rng.randint(0, n-1)
        current_tour[a], current_tour[b] = current_tour[b], current_tour[a]
        budget_remaining -= 1

    return np.array(best_tour)
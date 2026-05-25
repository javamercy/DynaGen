import numpy as np
import random

def solve_tsp(distance_matrix, seed, budget):
    n = distance_matrix.shape[0]
    rng = random.Random(seed)

    if n <= 2:
        tour = list(range(n))
        report_best_tour(tour)
        return np.array(tour)

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

    def compute_tour_dist(t, mat):
        d = 0
        for i in range(n):
            d += mat[t[i], t[(i+1)%n]]
        return d

    def two_opt(t, budget_left):
        improve = True
        while improve and budget_left > 0:
            improve = False
            for i in range(n-2):
                for j in range(i+2, n):  # j up to n-1, but we need j+1 mod n
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

        # Perturbation: double-bridge
        # Choose three random breakpoints and ensure non-empty segments
        a = rng.randint(0, n-1)
        b = rng.randint(0, n-1)
        c = rng.randint(0, n-1)
        # sort them
        i, j, k = sorted([a, b, c])
        # Adjust to ensure segments have at least one node? Not strictly required but may produce weaker perturbations.
        # We'll allow any; validity is guaranteed.
        seg1 = current_tour[:i+1]
        seg2 = current_tour[i+1:j+1]
        seg3 = current_tour[j+1:k+1]
        seg4 = current_tour[k+1:]
        current_tour = seg1 + seg3 + seg2 + seg4
        budget_remaining -= 1  # count perturbation as one unit

    return np.array(best_tour)
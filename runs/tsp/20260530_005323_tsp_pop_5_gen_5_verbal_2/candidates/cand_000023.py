import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    # farthest pair
    max_dist = -1
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i][j]
            if d > max_dist:
                max_dist = d
                best_pair = (i, j)
    tour = list(best_pair)
    unvisited = set(range(n)) - set(tour)
    while unvisited:
        best_regret = -1
        best_city = None
        best_pos = None
        for k in unvisited:
            costs = []
            L = len(tour)
            for pos in range(L):
                i = tour[pos]
                j = tour[(pos+1)%L]
                cost = distance_matrix[i][k] + distance_matrix[k][j] - distance_matrix[i][j]
                costs.append(cost)
            sorted_costs = sorted(costs)
            best = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs)>1 else float('inf')
            regret = second_best - best
            if regret > best_regret:
                best_regret = regret
                best_city = k
                best_pos = costs.index(best)
        tour.insert(best_pos+1, best_city)
        unvisited.remove(best_city)
    best_tour = tour[:]
    best_dist = float('inf')
    def tour_dist(t):
        d = 0.0
        for idx in range(n):
            d += distance_matrix[t[idx]][t[(idx+1)%n]]
        return d
    best_dist = tour_dist(best_tour)
    report_best_tour(np.array(best_tour))
    def two_opt(t):
        nonlocal best_dist, best_tour
        improved = True
        while improved:
            improved = False
            L = len(t)
            for i in range(L):
                for j in range(i+2, L):
                    a = t[i]
                    b = t[(i+1)%L]
                    c = t[j%L]
                    d = t[(j+1)%L]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old:
                        t[i+1:j+1] = reversed(t[i+1:j+1])
                        improved = True
                        new_dist = tour_dist(t)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_tour = t[:]
                            report_best_tour(np.array(t))
        return t
    tour = two_opt(tour)
    # perturbation loop (reduced iterations)
    for _ in range(5):
        L = len(tour)
        i = random.randint(1, L-4)
        j = random.randint(i+1, L-3)
        k = random.randint(j+1, L-2)
        new_tour = tour[:i] + tour[k:] + tour[j:k] + tour[i:j]
        new_tour = two_opt(new_tour)
        new_dist = tour_dist(new_tour)
        if new_dist < best_dist:
            best_dist = new_dist
            best_tour = new_tour[:]
            report_best_tour(np.array(new_tour))
        tour = new_tour
    return np.array(best_tour)
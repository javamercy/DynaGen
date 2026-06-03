import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # Regret-2 construction
    unvisited = set(range(n))
    i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = [i, j]
    unvisited.remove(i)
    unvisited.remove(j)
    best_costs = {}
    second_costs = {}
    best_positions = {}
    for city in unvisited:
        best = float('inf')
        second = float('inf')
        pos = 0
        for k in range(len(tour)):
            a = tour[k]
            b = tour[(k+1) % len(tour)]
            cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
            if cost < best:
                second = best
                best = cost
                pos = k+1
            elif cost < second:
                second = cost
        best_costs[city] = best
        second_costs[city] = second if second != float('inf') else best
        best_positions[city] = pos
    while unvisited:
        best_city = None
        max_regret = -float('inf')
        min_best = float('inf')
        for city in unvisited:
            regret = second_costs[city] - best_costs[city]
            if regret > max_regret or (regret == max_regret and best_costs[city] < min_best):
                max_regret = regret
                min_best = best_costs[city]
                best_city = city
        pos = best_positions[best_city]
        tour.insert(pos, best_city)
        unvisited.remove(best_city)
        for city in unvisited:
            best = float('inf')
            second = float('inf')
            pos = 0
            for k in range(len(tour)):
                a = tour[k]
                b = tour[(k+1) % len(tour)]
                cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                if cost < best:
                    second = best
                    best = cost
                    pos = k+1
                elif cost < second:
                    second = cost
            best_costs[city] = best
            second_costs[city] = second if second != float('inf') else best
            best_positions[city] = pos
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    # 2-opt local search
    def two_opt(tour):
        n = len(tour)
        improved = True
        best_tour = tour.copy()
        best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = best_tour[i]
                    b = best_tour[i+1]
                    c = best_tour[j]
                    d = best_tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        new_tour = best_tour.copy()
                        new_tour[i+1:j+1] = best_tour[j:i:-1]
                        new_dist = best_dist + delta
                        best_tour = new_tour
                        best_dist = new_dist
                        improved = True
                        report_best_tour(best_tour)
        return best_tour, best_dist
    tour_arr, best_dist = two_opt(tour_arr)
    # ILS with double-bridge perturbation and simulated annealing acceptance
    T0 = 0.1 * best_dist
    for iteration in range(n):
        # double-bridge perturbation
        i1 = np.random.randint(1, n//4)
        i2 = i1 + np.random.randint(1, n//4)
        i3 = i2 + np.random.randint(1, n//4)
        segment1 = tour_arr[:i1]
        segment2 = tour_arr[i1:i2]
        segment3 = tour_arr[i2:i3]
        segment4 = tour_arr[i3:]
        perturbed = np.concatenate([segment1, segment3, segment2, segment4])
        perturbed, new_dist = two_opt(perturbed)
        T = T0 * (1 - iteration / n)
        if new_dist < best_dist or random.random() < np.exp(-(new_dist - best_dist) / max(T, 1e-10)):
            tour_arr = perturbed
            best_dist = new_dist
            report_best_tour(tour_arr)
    return tour_arr
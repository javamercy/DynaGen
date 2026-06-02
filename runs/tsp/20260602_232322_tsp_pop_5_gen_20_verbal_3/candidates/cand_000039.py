import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    # Initial construction: random triangle + regret insertion
    start = random.sample(range(n), 3)
    tour = list(start)
    remaining = set(range(n)) - set(start)
    
    def delta(city, pos):
        before = tour[pos-1]
        after = tour[pos] if pos < len(tour) else tour[0]
        return distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
    
    while remaining:
        best_city = -1
        best_regret = -1
        best_pos = -1
        best_cost = float('inf')
        for city in remaining:
            costs = []
            for pos in range(len(tour)):
                costs.append((delta(city, pos), pos))
            costs.sort(key=lambda x: x[0])
            first = costs[0][0]
            second = costs[1][0] if len(costs) > 1 else first
            regret = second - first
            if regret > best_regret or (regret == best_regret and city < best_city):
                best_regret = regret
                best_city = city
                best_pos = costs[0][1]
                best_cost = first
        tour.insert(best_pos, best_city)
        remaining.remove(best_city)
    
    tour_arr = np.array(tour)
    best_tour = tour_arr.copy()
    best_dist = 0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    
    # Local search: iterate 2-opt and node insertion until no improvement
    improved = True
    while improved:
        improved = False
        # 2-opt
        for i in range(n-1):
            for j in range(i+2, n):
                if j - i < n:
                    new_tour = tour.copy()
                    new_tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    old = distance_matrix[a,b] + distance_matrix[c,d]
                    new = distance_matrix[a,c] + distance_matrix[b,d]
                    if new < old - 1e-10:
                        tour = new_tour
                        dist = 0
                        for i2 in range(n):
                            dist += distance_matrix[tour[i2], tour[(i2+1)%n]]
                        if dist < best_dist - 1e-10:
                            best_dist = dist
                            best_tour = np.array(tour)
                            report_best_tour(best_tour)
                        improved = True
                        break
            if improved:
                break
        if improved:
            continue
        # Node insertion (move each node to best position)
        for i in range(n):
            city = tour[i]
            best_pos = i
            best_delta = 0
            for j in range(n):
                if j == i:
                    continue
                # remove city from i, insert before j
                # compute delta
                prev_i = tour[(i-1)%n]
                next_i = tour[(i+1)%n]
                prev_j = tour[(j-1)%n]
                # new edges after removal: prev_i - next_i
                # new edges after insertion: prev_j - city, city - tour[j]
                # old edges: prev_i - city, city - next_i, prev_j - tour[j]
                delta_val = (distance_matrix[prev_i, next_i] + distance_matrix[prev_j, city] + distance_matrix[city, tour[j]]) - (distance_matrix[prev_i, city] + distance_matrix[city, next_i] + distance_matrix[prev_j, tour[j]])
                if delta_val < best_delta - 1e-10:
                    best_delta = delta_val
                    best_pos = j
            if best_pos != i and best_delta < -1e-10:
                # perform move
                city = tour.pop(i)
                if best_pos < i:
                    tour.insert(best_pos, city)
                else:
                    tour.insert(best_pos, city)
                dist = 0
                for i2 in range(n):
                    dist += distance_matrix[tour[i2], tour[(i2+1)%n]]
                if dist < best_dist - 1e-10:
                    best_dist = dist
                    best_tour = np.array(tour)
                    report_best_tour(best_tour)
                improved = True
                break
        if improved:
            continue
        break
    return best_tour
import numpy as np
import random
import math

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    tour = list(range(n))
    random.shuffle(tour)
    best_tour = tour.copy()
    best_cost = total_distance(tour, distance_matrix)
    report_best_tour(np.array(best_tour))
    T = 10.0
    T_min = 0.001
    alpha = 0.999
    max_iter = 10000
    iter = 0
    while T > T_min and iter < max_iter:
        # 2-opt move: select two indices to reverse segment
        i, j = random.sample(range(n), 2)
        if i > j:
            i, j = j, i
        if j - i == 1 or (i == 0 and j == n-1):
            continue
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        delta = total_distance(new_tour, distance_matrix) - total_distance(tour, distance_matrix)
        if delta < 0 or random.random() < math.exp(-delta / T):
            tour = new_tour
            current_cost = total_distance(tour, distance_matrix)
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = tour.copy()
                report_best_tour(np.array(best_tour))
        T *= alpha
        iter += 1
    return np.array(best_tour)

def total_distance(tour, dist):
    return sum(dist[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))
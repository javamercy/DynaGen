import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    # Nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda x: (distance_matrix[current][x], x))
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour, dtype=int)
    best_tour = tour.copy()
    best_cost = total_distance(distance_matrix, tour)
    report_best_tour(best_tour)
    
    # Adaptive threshold schedule
    epsilon_start = 0.1
    epsilon_end = 0.0
    max_passes = 10
    for pass_idx in range(max_passes):
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (pass_idx / max_passes)
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                new_tour = tour.copy()
                new_tour[i+1:j+1] = tour[j:i:-1]
                new_cost = total_distance(distance_matrix, new_tour)
                if new_cost < best_cost * (1 - epsilon):
                    best_cost = new_cost
                    best_tour = new_tour.copy()
                    tour = best_tour.copy()
                    report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
        if not improved and epsilon <= 0:
            break
    return best_tour

def total_distance(dist_matrix, tour):
    n = len(tour)
    total = 0.0
    for i in range(n - 1):
        total += dist_matrix[tour[i]][tour[i+1]]
    return total
import numpy as np
import math
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = list(range(n))
        try:
            report_best_tour(np.array(tour))
        except:
            pass
        return np.array(tour)
    
    random.seed(seed)
    # Nearest neighbor initial tour
    start = random.randrange(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    current = start
    while unvisited:
        nearest_city = min(unvisited, key=lambda c: distance_matrix[current][c])
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current = nearest_city
    
    best_tour = list(tour)
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i]][tour[(i+1)%n]]
    try:
        report_best_tour(np.array(best_tour))
    except:
        pass
    
    # Simulated annealing parameters
    # Budget: number of SA steps
    total_steps = max(1, budget // (n * 2))  # heuristic
    if total_steps < 10:
        total_steps = max(1, budget // 10)
    
    # Compute max distance for initial temperature
    max_dist = np.max(distance_matrix)
    T0 = max_dist * 0.3
    T = T0
    alpha = 0.99
    
    current_dist = best_dist
    current_tour = list(tour)
    
    for step in range(total_steps):
        if T < 1e-8:
            break
        # Generate random 2-opt move
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if j - i < 2 or (i == 0 and j == n-1):
            continue
        # Compute delta
        a, b = current_tour[i], current_tour[(i+1)%n]
        c, d = current_tour[j], current_tour[(j+1)%n]
        delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
        
        if delta < 0:
            # Accept
            current_tour[i+1:j+1] = reversed(current_tour[i+1:j+1])
            current_dist += delta
            if current_dist < best_dist:
                best_dist = current_dist
                best_tour = list(current_tour)
                try:
                    report_best_tour(np.array(best_tour))
                except:
                    pass
        else:
            # Accept with probability
            if random.random() < math.exp(-delta / T):
                current_tour[i+1:j+1] = reversed(current_tour[i+1:j+1])
                current_dist += delta
                # Note: if accepted worse, best remains unchanged
        T *= alpha
    
    return np.array(best_tour)
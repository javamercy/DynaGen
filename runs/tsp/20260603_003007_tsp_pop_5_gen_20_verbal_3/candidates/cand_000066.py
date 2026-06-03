import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    
    def compute_cost(tour):
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total
    
    # Nearest neighbor initialization
    unvisited = set(range(1, n))
    tour = [0]
    curr = 0
    while unvisited:
        next_city = min(unvisited, key=lambda city: distance_matrix[curr, city])
        tour.append(next_city)
        unvisited.remove(next_city)
        curr = next_city
    
    best_tour = tour.copy()
    best_cost = compute_cost(tour)
    report_best_tour(np.array(best_tour))
    
    temp = 100.0
    cooling = 0.999
    max_iterations = 20000
    no_improve_limit = 1500
    no_improve = 0
    
    for _ in range(max_iterations):
        # Random node insertion move
        i = random.randint(0, n-1)
        city = tour[i]
        new_tour = tour[:i] + tour[i+1:]
        j = random.randint(0, n-1)
        new_tour.insert(j, city)
        
        curr_cost = compute_cost(tour)
        new_cost = compute_cost(new_tour)
        delta = new_cost - curr_cost
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            tour = new_tour
            if new_cost < best_cost:
                best_cost = new_cost
                best_tour = tour.copy()
                report_best_tour(np.array(best_tour))
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Restart if stuck
        if no_improve >= no_improve_limit:
            # Perturb the best tour by 10 random insertions
            tour = best_tour.copy()
            for _ in range(10):
                i = random.randint(0, n-1)
                city = tour[i]
                tour = tour[:i] + tour[i+1:]
                j = random.randint(0, n-1)
                tour.insert(j, city)
            temp = 100.0  # Reset temperature
            no_improve = 0
        else:
            temp *= cooling
    
    return np.array(best_tour)
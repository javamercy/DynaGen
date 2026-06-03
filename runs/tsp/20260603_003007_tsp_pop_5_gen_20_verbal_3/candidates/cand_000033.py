import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    
    def nearest_neighbor(start):
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        curr = start
        while unvisited:
            next_city = min(unvisited, key=lambda city: distance_matrix[curr, city])
            tour.append(next_city)
            unvisited.remove(next_city)
            curr = next_city
        return tour
    
    def cost(tour):
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total
    
    # Initial tour
    current_tour = nearest_neighbor(0)
    current_cost = cost(current_tour)
    best_tour = current_tour.copy()
    best_cost = current_cost
    report_best_tour(np.array(best_tour))
    
    temp = 100.0
    cooling = 0.995
    max_iter = 10000
    restart_threshold = 2000
    no_improve = 0
    
    for _ in range(max_iter):
        # Insertion move
        i = random.randint(0, n-1)
        city = current_tour[i]
        new_tour = current_tour[:i] + current_tour[i+1:]
        j = random.randint(0, n-1)
        new_tour.insert(j, city)
        new_cost = cost(new_tour)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_tour = new_tour
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_tour = current_tour.copy()
                report_best_tour(np.array(best_tour))
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        if no_improve >= restart_threshold:
            # Restart from perturbed NN
            start = random.randint(0, n-1)
            current_tour = nearest_neighbor(start)
            current_cost = cost(current_tour)
            temp = 100.0  # reset temperature
            no_improve = 0
        else:
            temp *= cooling
    
    return np.array(best_tour)
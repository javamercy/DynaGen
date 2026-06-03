import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    # Nearest neighbor construction
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda city: distance_matrix[current, city])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    best_tour = tour.copy()
    best_cost = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n-1)) + distance_matrix[tour[-1], tour[0]]
    report_best_tour(np.array(best_tour))
    
    # Simulated annealing
    def cost(t):
        total = 0
        for i in range(n-1):
            total += distance_matrix[t[i], t[i+1]]
        total += distance_matrix[t[-1], t[0]]
        return total
    
    temp = 100.0
    cooling_rate = 0.995
    iterations = 10000
    for _ in range(iterations):
        # 2-opt move: reverse a subsequence
        i, j = sorted(random.sample(range(n), 2))
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        current_cost = cost(tour)
        new_cost = cost(new_tour)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / temp):
            tour = new_tour
            if new_cost < best_cost:
                best_cost = new_cost
                best_tour = tour.copy()
                report_best_tour(np.array(best_tour))
        temp *= cooling_rate
    return np.array(best_tour)
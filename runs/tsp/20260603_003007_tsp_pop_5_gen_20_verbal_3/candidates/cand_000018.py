import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    # Nearest neighbor construction
    unvisited = set(range(1, n))
    tour = [0]
    curr = 0
    while unvisited:
        next_city = min(unvisited, key=lambda city: distance_matrix[curr, city])
        tour.append(next_city)
        unvisited.remove(next_city)
        curr = next_city
    best_tour = tour.copy()
    best_cost = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n-1)) + distance_matrix[tour[-1], tour[0]]
    report_best_tour(np.array(best_tour))

    def cost(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1)%n]]
        return total

    temp = 100.0
    cooling = 0.995
    iterations = 10000
    for _ in range(iterations):
        # Random node insertion move
        i = random.randint(0, n-1)
        city = tour[i]
        new_tour = tour[:i] + tour[i+1:]
        j = random.randint(0, n-1)
        new_tour.insert(j, city)
        curr_cost = cost(tour)
        new_cost = cost(new_tour)
        delta = new_cost - curr_cost
        if delta < 0 or random.random() < np.exp(-delta / temp):
            tour = new_tour
            if new_cost < best_cost:
                best_cost = new_cost
                best_tour = tour.copy()
                report_best_tour(np.array(best_tour))
        temp *= cooling
    return np.array(best_tour)
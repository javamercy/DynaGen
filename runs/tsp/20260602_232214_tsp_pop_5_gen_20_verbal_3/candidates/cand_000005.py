import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    best_tour = None
    best_cost = float('inf')
    
    def tour_cost(tour):
        cost = 0.0
        for i in range(n):
            cost += distance_matrix[tour[i], tour[(i+1)%n]]
        return cost
    
    def two_opt(tour, cost):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    if j == n-1 and i == 0:
                        continue
                    i1 = (i+1)%n
                    j1 = (j+1)%n
                    delta = -distance_matrix[tour[i], tour[i1]] - distance_matrix[tour[j], tour[j1]] + distance_matrix[tour[i], tour[j]] + distance_matrix[tour[i1], tour[j1]]
                    if delta < -1e-10:
                        tour[i1:j+1] = tour[i1:j+1][::-1]
                        cost += delta
                        improved = True
                        if cost < best_cost:
                            best_cost = cost
                            report_best_tour(tour.copy())
        return tour, cost
    
    for restart in range(10):
        unvisited = set(range(n))
        start = random.randint(0, n-1)
        tour = [start]
        unvisited.remove(start)
        current = start
        while unvisited:
            dists = [(distance_matrix[current, v], v) for v in unvisited]
            dists.sort()
            k = min(5, len(dists))
            candidates = dists[:k]
            _, next_city = random.choice(candidates)
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        tour = np.array(tour, dtype=int)
        cost = tour_cost(tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)
        improved_tour, improved_cost = two_opt(tour, cost)
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_tour = improved_tour.copy()
            report_best_tour(best_tour)
    return best_tour
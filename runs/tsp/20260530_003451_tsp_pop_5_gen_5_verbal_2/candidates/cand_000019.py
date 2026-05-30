import numpy as np
import random

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    def tour_cost(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
    # Nearest neighbor
    tour = [0]
    unvisited = set(range(1, n))
    curr = 0
    while unvisited:
        nxt = min(unvisited, key=lambda c: distance_matrix[curr, c])
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    best_cost = tour_cost(best_tour)
    report_best_tour(best_tour)
    
    def two_opt(t):
        nonlocal best_cost, best_tour
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a, b = t[i], t[(i+1)%n]
                    c, d = t[j], t[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-12:
                        t[i+1:j+1] = t[i+1:j+1][::-1]
                        new_cost = tour_cost(t)
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_tour = t.copy()
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
        return t
    
    tour = two_opt(tour)
    # Restarts
    for _ in range(10):
        new_tour = best_tour.copy()
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i > j:
            i, j = j, i
        if j - i > 1:
            new_tour[i:j+1] = new_tour[i:j+1][::-1]
        else:
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_tour = two_opt(new_tour)
        cost = tour_cost(new_tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = new_tour.copy()
            report_best_tour(best_tour)
    return best_tour
import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = list(range(n))
        try:
            report_best_tour(tour)
        except:
            pass
        return np.array(tour)
    random.seed(seed)
    # Construction: cheapest insertion
    start = random.randrange(n)
    tour = [start]
    unvisited = list(set(range(n)) - {start})
    while unvisited:
        best_city = None
        best_cost = float('inf')
        best_pos = None
        for city in unvisited:
            for idx in range(len(tour)):
                if idx == len(tour) - 1:
                    cost = distance_matrix[tour[idx]][city] + distance_matrix[city][tour[0]] - distance_matrix[tour[idx]][tour[0]]
                else:
                    cost = distance_matrix[tour[idx]][city] + distance_matrix[city][tour[idx+1]] - distance_matrix[tour[idx]][tour[idx+1]]
                if cost < best_cost:
                    best_cost = cost
                    best_city = city
                    best_pos = idx
        tour.insert(best_pos + 1, best_city)
        unvisited.remove(best_city)
    try:
        report_best_tour(np.array(tour))
    except:
        pass
    # Improvement: Or-opt node relocation
    max_iter = max(1, budget // 10)
    iteration = 0
    improved = True
    while improved and iteration < max_iter:
        improved = False
        n_tour = len(tour)
        for i in range(n_tour):
            node = tour[i]
            # remove node
            new_tour = tour[:i] + tour[i+1:]
            best_delta = 0.0
            best_insert_pos = -1
            for j in range(len(new_tour) + 1):
                # compute delta cost
                prev = new_tour[j-1] if j > 0 else new_tour[-1]
                nxt = new_tour[j] if j < len(new_tour) else new_tour[0]
                old_prev = tour[i-1] if i > 0 else tour[-1]
                old_nxt = tour[i+1] if i+1 < n_tour else tour[0]
                # old edges: (old_prev, node) and (node, old_nxt)
                old_cost = distance_matrix[old_prev][node] + distance_matrix[node][old_nxt]
                # new edges: (prev, node) and (node, nxt)
                new_cost = distance_matrix[prev][node] + distance_matrix[node][nxt]
                delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_insert_pos = j
            if best_delta < 0:
                # apply move
                tour = new_tour[:best_insert_pos] + [node] + new_tour[best_insert_pos:]
                improved = True
                try:
                    report_best_tour(np.array(tour))
                except:
                    pass
                break  # start new pass after improvement
        iteration += 1
    return np.array(tour)
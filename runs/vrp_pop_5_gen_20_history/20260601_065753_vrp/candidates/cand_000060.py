import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def compute_route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0 for _ in range(truck_count)]
        for cust in perm:
            r = min(range(truck_count), key=lambda i: lengths[i])
            routes[r].insert(-1, cust)
            lengths[r] = compute_route_length(routes[r])
        return routes
    
    best_max = float('inf')
    best_routes = None
    
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
    
    max_trials = 100 * n  # finite bound
    for _ in range(max_trials):
        perm = customers[:]
        random.shuffle(perm)
        routes = decode(perm)
        report_best_vrp(routes)
    
    return best_routes
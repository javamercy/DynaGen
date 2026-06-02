import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count
    for cust in customers:
        best_new_max = float('inf')
        best_new_route_d = float('inf')
        best_route_idx = None
        best_pos = None
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                old_d = route_distances[r_idx]
                removed = distance_matrix[route[pos-1], route[pos]]
                added = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]]
                new_d = old_d - removed + added
                other_max = 0.0
                for j in range(truck_count):
                    if j != r_idx:
                        if route_distances[j] > other_max:
                            other_max = route_distances[j]
                new_max = max(new_d, other_max)
                if new_max < best_new_max or (new_max == best_new_max and new_d < best_new_route_d):
                    best_new_max = new_max
                    best_new_route_d = new_d
                    best_route_idx = r_idx
                    best_pos = pos
        routes[best_route_idx].insert(best_pos, cust)
        route_distances[best_route_idx] = best_new_route_d
    report_best_vrp(routes)
    return routes
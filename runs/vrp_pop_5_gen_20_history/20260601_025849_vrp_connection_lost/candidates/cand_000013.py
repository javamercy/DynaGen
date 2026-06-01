import numpy as np
import math
import heapq
import itertools
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    
    # Helper: route distance
    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d
    
    def max_dist(routes):
        return max(route_dist(r) for r in routes)
    
    # Regret-2 construction with tie-breaking by max distance impact
    while unassigned:
        best_info = {}
        for c in unassigned:
            best = float('inf')
            second = float('inf')
            best_r = -1
            best_p = -1
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                    if cost < best:
                        second = best
                        best = cost
                        best_r = r_idx
                        best_p = i + 1
                    elif cost < second:
                        second = cost
            best_info[c] = (best, second, best_r, best_p)
        
        candidates = []
        for c, (best, second, r_idx, pos) in best_info.items():
            regret = second - best if second != float('inf') else float('inf')
            new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
            new_route_dist = route_dist(new_route)
            other_max = max(route_dist(r) for i, r in enumerate(routes) if i != r_idx) if truck_count > 1 else 0.0
            new_max = max(new_route_dist, other_max)
            candidates.append((-regret, new_max, c, r_idx, pos))
        
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_c, chosen_r, chosen_p = candidates[0]
        routes[chosen_r].insert(chosen_p, chosen_c)
        unassigned.remove(chosen_c)
    
    report_best_vrp(routes)
    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)
    
    # Bounded swap improvement
    max_iter = n * truck_count
    for _ in range(max_iter):
        improved = False
        # iterate over all pairs of routes and positions
        for r1 in range(truck_count):
            for pos1 in range(1, len(routes[r1])-1):
                c1 = routes[r1][pos1]
                for r2 in range(r1+1, truck_count):
                    for pos2 in range(1, len(routes[r2])-1):
                        c2 = routes[r2][pos2]
                        new_r1 = routes[r1][:pos1] + [c2] + routes[r1][pos1+1:]
                        new_r2 = routes[r2][:pos2] + [c1] + routes[r2][pos2+1:]
                        new_routes = [list(r) for r in routes]
                        new_routes[r1] = new_r1
                        new_routes[r2] = new_r2
                        new_max = max_dist(new_routes)
                        if new_max < best_max - 1e-9:
                            best_max = new_max
                            best_routes = [list(r) for r in new_routes]
                            routes = new_routes
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # Ensure exactly truck_count routes, each starting and ending at 0
    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    return final_routes
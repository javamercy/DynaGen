import numpy as np
import random
from typing import List

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route: List[int]) -> float:
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def max_route_distance(routes: List[List[int]]) -> float:
        return max(route_distance(r) for r in routes)
    
    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    
    # List of customers (1..n-1)
    customers = list(range(1, n))
    random.shuffle(customers)  # add randomness to avoid bias
    
    for cust in customers:
        best_increase = float('inf')
        best_route = -1
        best_pos = -1
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                # new route if we insert here
                new_route = route[:pos] + [cust] + route[pos:]
                old_dist = route_distance(route)
                new_dist = route_distance(new_route)
                new_max = max(new_dist, max(route_distance(r) for r in routes if r is not route))
                increase = new_max - max_route_distance(routes)
                if increase < best_increase:
                    best_increase = increase
                    best_route = r_idx
                    best_pos = pos
        routes[best_route].insert(best_pos, cust)
    
    best_routes = [r[:] for r in routes]
    best_max = max_route_distance(best_routes)
    report_best_vrp(best_routes)
    
    # Local search: inter-route relocate (first improvement)
    max_iter_relocate = (n - 1) * truck_count * 5
    for _ in range(max_iter_relocate):
        improved = False
        for src in range(truck_count):
            route_src = routes[src]
            if len(route_src) <= 2:
                continue
            for pos_src in range(1, len(route_src)-1):
                cust = route_src[pos_src]
                temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                dist_src = route_distance(temp_src)
                for dst in range(truck_count):
                    if dst == src:
                        continue
                    route_dst = routes[dst]
                    for pos_dst in range(1, len(route_dst)):
                        new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                        dist_dst = route_distance(new_dst)
                        new_max = max(dist_src, dist_dst, max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst))
                        if new_max < best_max:
                            routes[src] = temp_src
                            routes[dst] = new_dst
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
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
    
    # Intra-route 2-opt
    max_iter_2opt = (n - 1) * 10
    for _ in range(max_iter_2opt):
        improved = False
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    old_dist = route_distance(route)
                    new_dist = route_distance(new_route)
                    if new_dist >= old_dist:
                        continue
                    new_max = max(new_dist, max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx))
                    if new_max < best_max:
                        routes[r_idx] = new_route
                        best_max = new_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_routes
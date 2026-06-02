import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    unvisited = set(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_dist = [0.0 for _ in range(truck_count)]
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    # Initial insertion of all customers using cheapest insertion
    while unvisited:
        best_customer = None
        best_route_idx = None
        best_pos = None
        best_inc = float('inf')
        for cust in sorted(unvisited):
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                    if inc < best_inc - 1e-9:
                        best_inc = inc
                        best_customer = cust
                        best_route_idx = r_idx
                        best_pos = pos
        if best_customer is not None:
            routes[best_route_idx].insert(best_pos, best_customer)
            unvisited.remove(best_customer)
            route_dist[best_route_idx] = route_distance(routes[best_route_idx])
    
    # Improvement: relocate from longest route to reduce max distance
    max_iter = n * truck_count
    best_routes = [list(r) for r in routes]
    best_max_dist = max(route_dist)
    report_best_vrp(best_routes)
    
    for _ in range(max_iter):
        max_dist = max(route_dist)
        long_indices = [i for i, d in enumerate(route_dist) if abs(d - max_dist) < 1e-9]
        if not long_indices:
            break
        improved = False
        for long_idx in long_indices:
            route = routes[long_idx]
            if len(route) <= 2:
                continue
            customers_with_savings = []
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                nxt = route[pos+1]
                savings = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                customers_with_savings.append((cust, pos, savings))
            customers_with_savings.sort(key=lambda x: -x[2])
            for cust, pos, savings in customers_with_savings:
                new_route_long = route[:pos] + route[pos+1:]
                dist_long_new = route_distance(new_route_long)
                for short_idx, short_route in enumerate(routes):
                    if short_idx == long_idx:
                        continue
                    for p in range(1, len(short_route)):
                        prev = short_route[p-1]
                        nxt = short_route[p]
                        inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                        new_short_route = short_route[:p] + [cust] + short_route[p:]
                        dist_short_new = route_distance(new_short_route)
                        new_max = max(dist_long_new, dist_short_new, max(d for i,d in enumerate(route_dist) if i not in (long_idx, short_idx)))
                        if new_max < best_max_dist - 1e-9:
                            routes[long_idx] = new_route_long
                            routes[short_idx] = new_short_route
                            route_dist[long_idx] = dist_long_new
                            route_dist[short_idx] = dist_short_new
                            best_max_dist = new_max
                            best_routes = [list(r) for r in routes]
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
    
    # Inter-route 2-opt improvement (swap tails)
    for _ in range(max_iter):
        max_dist = max(route_dist)
        long_indices = [i for i, d in enumerate(route_dist) if abs(d - max_dist) < 1e-9]
        if not long_indices:
            break
        improved = False
        for long_idx in long_indices:
            route1 = routes[long_idx]
            if len(route1) <= 2:
                continue
            for other_idx in range(truck_count):
                if other_idx == long_idx:
                    continue
                route2 = routes[other_idx]
                if len(route2) <= 2:
                    continue
                # Try all cut positions
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        # new routes after swapping tails
                        new_route1 = route1[:i+1] + route2[j+1:]
                        new_route2 = route2[:j+1] + route1[i+1:]
                        # check that no duplicate customers (depot 0 might appear twice)
                        # but swapping tails keeps each customer exactly once since segments are disjoint
                        # However, we must ensure that the first segment ends with a customer not depot? It's fine.
                        d1 = route_distance(new_route1)
                        d2 = route_distance(new_route2)
                        new_max = max(d1, d2, max(d for idx,d in enumerate(route_dist) if idx not in (long_idx, other_idx)))
                        if new_max < best_max_dist - 1e-9:
                            routes[long_idx] = new_route1
                            routes[other_idx] = new_route2
                            route_dist[long_idx] = d1
                            route_dist[other_idx] = d2
                            best_max_dist = new_max
                            best_routes = [list(r) for r in routes]
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
    
    return best_routes
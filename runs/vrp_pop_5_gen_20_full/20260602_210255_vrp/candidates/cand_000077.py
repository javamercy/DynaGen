import numpy as np
import random

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
    
    # Construction: cheapest insertion
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
    
    best_routes = [list(r) for r in routes]
    best_max_dist = max(route_dist)
    report_best_vrp(best_routes)
    
    max_iter = 5 * n * truck_count  # bounded
    iteration_no_improve = 0
    max_no_improve = n
    max_perturbations = 5
    perturbations_done = 0
    
    for _ in range(max_iter):
        if perturbations_done >= max_perturbations:
            break
        max_dist = max(route_dist)
        long_indices = [i for i, d in enumerate(route_dist) if abs(d - max_dist) < 1e-9]
        improved = False
        for long_idx in long_indices:
            route = routes[long_idx]
            if len(route) <= 2:
                continue
            # Build list of customers with removal savings descending
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
                            # improving move
                            routes[long_idx] = new_route_long
                            routes[short_idx] = new_short_route
                            route_dist[long_idx] = dist_long_new
                            route_dist[short_idx] = dist_short_new
                            best_max_dist = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            iteration_no_improve = 0
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            iteration_no_improve += 1
            if iteration_no_improve >= max_no_improve and perturbations_done < max_perturbations:
                # Perturbation: randomly relocate a customer from longest route to another route
                perturbations_done += 1
                long_idx = max(range(truck_count), key=lambda i: route_dist[i])
                route = routes[long_idx]
                if len(route) > 2:
                    # pick random customer from longest route
                    pos = random.randint(1, len(route)-2)
                    cust = route[pos]
                    new_route_long = route[:pos] + route[pos+1:]
                    dist_long_new = route_distance(new_route_long)
                    # choose random target route
                    short_idx = random.choice([i for i in range(truck_count) if i != long_idx])
                    short_route = routes[short_idx]
                    p = random.randint(1, len(short_route)-1)
                    new_short_route = short_route[:p] + [cust] + short_route[p:]
                    dist_short_new = route_distance(new_short_route)
                    routes[long_idx] = new_route_long
                    routes[short_idx] = new_short_route
                    route_dist[long_idx] = dist_long_new
                    route_dist[short_idx] = dist_short_new
                    new_max = max(route_dist)
                    if new_max < best_max_dist:
                        best_max_dist = new_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                    iteration_no_improve = 0
    return best_routes
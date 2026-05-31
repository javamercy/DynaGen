import numpy as np
import math
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= n - 1:
        routes = []
        for i in customers:
            routes.append([0, i, 0])
        for _ in range(truck_count - len(customers)):
            routes.append([0, 0])
        return routes
    
    # Deterministic seed selection: choose customers farthest from depot, tie-break by index
    dist_from_depot = [(distance_matrix[0][i], i) for i in range(1, n)]
    dist_from_depot.sort(key=lambda x: (-x[0], x[1]))
    seeds = [x[1] for x in dist_from_depot[:truck_count]]
    
    # Cluster assignment: each customer to nearest seed, tie-break by smaller seed index
    clusters = {seed: [] for seed in seeds}
    for cust in customers:
        best_seed = min(seeds, key=lambda s: (distance_matrix[cust][s], s))
        clusters[best_seed].append(cust)
    
    def compute_route_distance(route):
        if len(route) == 2:
            return 0.0
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        return dist
    
    def nearest_neighbor_route(nodes):
        if not nodes:
            return [0, 0]
        route = [0]
        remaining = set(nodes)
        current = 0
        while remaining:
            next_node = min(remaining, key=lambda x: distance_matrix[current][x])
            route.append(next_node)
            remaining.remove(next_node)
            current = next_node
        route.append(0)
        return route
    
    def two_opt_improve(route):
        if len(route) <= 4:
            return route
        improved = True
        max_iter = len(route) * 10
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            for i in range(1, len(route)-3):
                for j in range(i+1, len(route)-2):
                    if j-i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = compute_route_distance(new_route)
                    old_dist = compute_route_distance(route)
                    if new_dist < old_dist:
                        route = new_route
                        improved = True
        return route
    
    # Build initial routes
    routes = []
    for seed in seeds:
        clist = clusters[seed]
        route = nearest_neighbor_route(clist)
        route = two_opt_improve(route)
        routes.append(route)
    
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    best_routes = [route[:] for route in routes]
    best_max = max(compute_route_distance(r) for r in routes)
    
    # Helper to get best insertion position and increase for a customer into a route
    def best_insertion(route, cust):
        if len(route) == 2:
            # empty route
            new_route = [0, cust, 0]
            increase = 2 * distance_matrix[0][cust]
            return new_route, increase
        best_increase = float('inf')
        best_new_route = None
        for pos in range(1, len(route)):
            a = route[pos-1]
            b = route[pos]
            increase = distance_matrix[a][cust] + distance_matrix[cust][b] - distance_matrix[a][b]
            if increase < best_increase:
                best_increase = increase
                best_new_route = route[:pos] + [cust] + route[pos:]
        return best_new_route, best_increase
    
    n_cust = n - 1
    max_iter = min(100, n_cust * truck_count)
    for _ in range(max_iter):
        route_dists = [compute_route_distance(r) for r in routes]
        max_idx = max(range(truck_count), key=lambda i: (route_dists[i], i))
        min_idx = min(range(truck_count), key=lambda i: (route_dists[i], i))
        if max_idx == min_idx:
            break
        if route_dists[max_idx] == 0:
            break
        improved = False
        # move a customer from max route to min route
        max_route = routes[max_idx]
        if len(max_route) > 3:  # at least one customer
            customers_max = max_route[1:-1]
            customers_max.sort()  # deterministic order
            for cust in customers_max:
                new_min_route, increase = best_insertion(routes[min_idx], cust)
                new_max_route = [0] + [c for c in max_route[1:-1] if c != cust] + [0]
                new_max_dist = compute_route_distance(new_max_route)
                new_min_dist = compute_route_distance(new_min_route) if len(new_min_route) > 2 else 0
                # compute new max among all routes
                new_max = new_max_dist
                if new_min_dist > new_max:
                    new_max = new_min_dist
                for i in range(truck_count):
                    if i == max_idx or i == min_idx:
                        continue
                    if route_dists[i] > new_max:
                        new_max = route_dists[i]
                if new_max < best_max - 1e-9:
                    # apply move
                    routes[max_idx] = new_max_route
                    routes[min_idx] = new_min_route
                    best_max = new_max
                    best_routes = [route[:] for route in routes]
                    improved = True
                    break
        if not improved:
            # try swap a customer between max and min routes
            max_route = routes[max_idx]
            min_route = routes[min_idx]
            if len(max_route) > 3 and len(min_route) > 3:
                customers_max = max_route[1:-1]
                customers_min = min_route[1:-1]
                customers_max.sort()
                customers_min.sort()
                for cust_max in customers_max:
                    if improved:
                        break
                    for cust_min in customers_min:
                        # remove cust_max from max, add cust_min; remove cust_min from min, add cust_max
                        new_max_route = [0] + [c for c in max_route[1:-1] if c != cust_max] + [cust_min] + [0]
                        new_min_route = [0] + [c for c in min_route[1:-1] if c != cust_min] + [cust_max] + [0]
                        # we need to reorder these routes (nearest neighbor?) but for simplicity compute distance directly
                        # Actually we need to re-optimize the routes after swap, but for evaluation compute distance as is (not optimal)
                        # Better to compute distance of the route as a sequence (the order matters). We'll use the same order as insertion at end? That might be suboptimal. Instead, we can compute distance for the new routes as the sum of edges in the given order.
                        # But that might increase distance erroneously. Since we are not reordering, swap might look bad. We'll compute distance for the route as is.
                        new_max_dist = compute_route_distance(new_max_route)
                        new_min_dist = compute_route_distance(new_min_route)
                        new_max = max(new_max_dist, new_min_dist, max(route_dists[i] for i in range(truck_count) if i != max_idx and i != min_idx))
                        if new_max < best_max - 1e-9:
                            # apply swap and then re-optimize both routes with 2-opt
                            new_max_route = two_opt_improve(new_max_route)
                            new_min_route = two_opt_improve(new_min_route)
                            new_max_dist = compute_route_distance(new_max_route)
                            new_min_dist = compute_route_distance(new_min_route)
                            new_max = max(new_max_dist, new_min_dist, max(route_dists[i] for i in range(truck_count) if i != max_idx and i != min_idx))
                            if new_max < best_max - 1e-9:
                                routes[max_idx] = new_max_route
                                routes[min_idx] = new_min_route
                                best_max = new_max
                                best_routes = [route[:] for route in routes]
                                improved = True
                                break
        if not improved:
            break
    
    # Re-optimize each route with 2-opt one final time
    for i in range(truck_count):
        routes[i] = two_opt_improve(routes[i])
    # Update best if improved
    current_max = max(compute_route_distance(r) for r in routes)
    if current_max < best_max:
        best_max = current_max
        best_routes = [route[:] for route in routes]
    
    # Ensure all routes start and end at 0
    for i, r in enumerate(best_routes):
        if r[0] != 0 or r[-1] != 0:
            best_routes[i] = [0] + r[1:-1] + [0] if len(r) > 2 else [0, 0]
    return best_routes

# Helper function (not required but used within):
# Already defined above.
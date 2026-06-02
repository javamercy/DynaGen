import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    best_routes = None
    best_max = float('inf')
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    # Number of restarts: fixed small number to keep runtime bounded
    restart_count = min(5, max(1, n // 20))
    
    for restart in range(restart_count):
        # Initialize empty routes
        routes = [[0, 0] for _ in range(truck_count)]
        # Random permutation of customers
        customers = list(range(1, n))
        random.shuffle(customers)
        
        # Cheapest insertion for each customer in that order
        for cust in customers:
            best_cost = float('inf')
            best_route = None
            best_pos = None
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    a = route[pos-1]
                    b = route[pos]
                    inc = distance_matrix[a, cust] + distance_matrix[cust, b] - distance_matrix[a, b]
                    if inc < best_cost:
                        best_cost = inc
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)
        
        # Current max distance
        current_max = max(route_dist(r) for r in routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        
        # Local search: bounded iterations
        max_iter = n * 2
        for it in range(max_iter):
            improved = False
            # Intra-route 2-opt
            for r_idx, route in enumerate(routes):
                if len(route) <= 2:
                    continue
                best_route_dist = route_dist(route)
                best_route = route[:]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_dist(new_route)
                        if new_dist < best_route_dist:
                            best_route_dist = new_dist
                            best_route = new_route
                            improved = True
                routes[r_idx] = best_route
            
            # Inter-route relocate: move a customer from longest route to another
            dists = [route_dist(r) for r in routes]
            current_max = max(dists)
            if current_max < best_max:
                best_max = current_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            # Find longest route
            longest_idx = max(range(truck_count), key=lambda i: dists[i])
            longest_route = routes[longest_idx]
            for cust_pos in range(1, len(longest_route)-1):
                cust = longest_route[cust_pos]
                # Try moving to other routes
                for other_idx in range(truck_count):
                    if other_idx == longest_idx:
                        continue
                    other_route = routes[other_idx]
                    best_inc = float('inf')
                    best_pos = None
                    for pos in range(1, len(other_route)):
                        a = other_route[pos-1]
                        b = other_route[pos]
                        inc = distance_matrix[a, cust] + distance_matrix[cust, b] - distance_matrix[a, b]
                        if inc < best_inc:
                            best_inc = inc
                            best_pos = pos
                    # Compute new max
                    new_long = longest_route[:cust_pos] + longest_route[cust_pos+1:]
                    new_other = other_route[:best_pos] + [cust] + other_route[best_pos:]
                    new_dists = [route_dist(new_long), route_dist(new_other)]
                    for k in range(truck_count):
                        if k not in (longest_idx, other_idx):
                            new_dists.append(dists[k])
                    new_max = max(new_dists)
                    if new_max < current_max:
                        routes[longest_idx] = new_long
                        routes[other_idx] = new_other
                        improved = True
                        current_max = new_max
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        break  # restart move search
                if improved:
                    break
            if improved:
                continue
            
            # Inter-route swap: swap two customers between two routes
            # Only consider pairs of routes where one is longest
            swap_improved = False
            for i in range(truck_count):
                for j in range(i+1, truck_count):
                    route_i = routes[i]
                    route_j = routes[j]
                    for pos_i in range(1, len(route_i)-1):
                        cust_i = route_i[pos_i]
                        for pos_j in range(1, len(route_j)-1):
                            cust_j = route_j[pos_j]
                            # Compute new routes
                            new_i = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
                            new_j = route_j[:pos_j] + [cust_i] + route_j[pos_j+1:]
                            new_dist_i = route_dist(new_i)
                            new_dist_j = route_dist(new_j)
                            # Current distances
                            dist_i = dists[i]
                            dist_j = dists[j]
                            # New max only depends on these two, but we need overall max
                            # Simpler: check if max of new_dist_i, new_dist_j, and other dists < current_max
                            other_dists = [dists[k] for k in range(truck_count) if k not in (i, j)]
                            new_max = max([new_dist_i, new_dist_j] + other_dists)
                            if new_max < current_max:
                                routes[i] = new_i
                                routes[j] = new_j
                                improved = True
                                swap_improved = True
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [list(r) for r in routes]
                                    report_best_vrp(best_routes)
                                break
                        if swap_improved:
                            break
                    if swap_improved:
                        break
                if swap_improved:
                    break
            if not improved:
                break
    
    return best_routes
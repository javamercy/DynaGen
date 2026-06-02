import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    
    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
    
    def objective(routes):
        return max(route_distance(r) for r in routes)
    
    # Helper to compute new route distance if inserting node at pos
    def insertion_cost(route, node, pos):
        # pos is insertion index (1..len-1)
        old = sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
        # new distance after insertion: remove edge (prev, next) and add (prev, node) and (node, next)
        prev = route[pos-1]
        nxt = route[pos]
        new = old - dist[prev][nxt] + dist[prev][node] + dist[node][nxt]
        return new
    
    # Initial solution using minimax construction (same as parent 1)
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    while unassigned:
        best_max = float('inf')
        best_total = float('inf')
        best_node = None
        best_route = None
        best_pos = None
        for node in unassigned:
            for r in range(truck_count):
                route = routes[r]
                for pos in range(1, len(route)):
                    new_dist = insertion_cost(route, node, pos)
                    current_max = route_distance(route) if r != 0 else 0  # will compute explicitly below
                    # Compute max across all routes considering new distance for route r
                    max_other = 0
                    for rr in range(truck_count):
                        if rr == r:
                            d = new_dist
                        else:
                            d = route_distance(routes[rr])
                        if d > max_other:
                            max_other = d
                    if max_other < best_max or (max_other == best_max and new_dist < best_total):
                        best_max = max_other
                        best_total = new_dist
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)
    
    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)
    
    # Ruin and recreate parameters
    max_iter = min(100, n * 2)
    all_customers = list(range(1, n))
    
    for _ in range(max_iter):
        # Backup current routes
        current_routes = [list(r) for r in routes]
        
        # Determine number of customers to remove (20-40%)
        n_remove = random.randint(max(1, int(0.2 * (n-1))), max(1, int(0.4 * (n-1))))
        
        # Compute route distances and sort routes by distance descending
        route_dists = [route_distance(r) for r in routes]
        sorted_route_indices = sorted(range(truck_count), key=lambda i: route_dists[i], reverse=True)
        
        # Collect customers to remove, biased towards longer routes
        removed = []
        remaining_to_remove = n_remove
        for ri in sorted_route_indices:
            route = routes[ri]
            if len(route) <= 2:
                continue
            # Remove up to half of the customers on this route (excluding depot)
            route_customers = route[1:-1]
            if len(route_customers) == 0:
                continue
            # Number to remove from this route: proportional to its length? simple: remove random subset
            max_remove_from_route = min(remaining_to_remove, len(route_customers))
            if max_remove_from_route <= 0:
                continue
            # Randomly choose customers from this route
            remove_from_route = random.sample(range(len(route_customers)), min(max_remove_from_route, len(route_customers)))
            for idx in sorted(remove_from_route, reverse=True):
                customer = route_customers[idx]
                removed.append(customer)
                # Remove from route
                pos = route.index(customer)
                routes[ri] = route[:pos] + route[pos+1:]
                if len(routes[ri]) < 2:
                    routes[ri] = [0, 0]
            remaining_to_remove -= len(remove_from_route)
            if remaining_to_remove <= 0:
                break
        # If still need to remove more, remove random from remaining customers
        if remaining_to_remove > 0:
            # Get all customers currently in routes
            in_routes = set()
            for r in routes:
                in_routes.update(r[1:-1])
            available = list(in_routes - set(removed))
            if len(available) > 0:
                extra = random.sample(available, min(remaining_to_remove, len(available)))
                for customer in extra:
                    removed.append(customer)
                    for ri, route in enumerate(routes):
                        if customer in route:
                            pos = route.index(customer)
                            routes[ri] = route[:pos] + route[pos+1:]
                            if len(routes[ri]) < 2:
                                routes[ri] = [0, 0]
                            break
        
        # Now reconstruct: order removed customers by distance to depot descending
        removed.sort(key=lambda node: dist[0][node], reverse=True)
        
        # Insert using minimax insertion
        unassigned = removed
        while unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_node = None
            best_route = None
            best_pos = None
            for node in unassigned:
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        new_dist = insertion_cost(route, node, pos)
                        max_other = 0
                        for rr in range(truck_count):
                            if rr == r:
                                d = new_dist
                            else:
                                d = route_distance(routes[rr])
                            if d > max_other:
                                max_other = d
                        if max_other < best_max or (max_other == best_max and new_dist < best_total):
                            best_max = max_other
                            best_total = new_dist
                            best_node = node
                            best_route = r
                            best_pos = pos
            routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)
        
        # Apply intra-route 2-opt on each route
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            improved = True
            local_iter = 0
            while improved and local_iter < 5:
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        old_dist = route_distance(route)
                        if new_dist + 1e-10 < old_dist:
                            route = new_route
                            routes[r] = route
                            improved = True
                            break
                    if improved:
                        break
                local_iter += 1
        
        new_obj = objective(routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        # Always accept new solution as current
        # (routes already updated)
    
    return best_routes
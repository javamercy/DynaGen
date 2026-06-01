import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(42)  # deterministic randomness
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def report_best(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]
    
    # trivial case
    if truck_count >= n-1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best(routes)
        return best_routes
    
    # Savings construction
    # Initialize each customer as a route
    routes = [[0, c, 0] for c in customers]
    # Each route has two ends: the first customer (index 1) and the last customer (index -2)
    # For each route, store ends: (first_customer, last_customer)
    route_ends = [(r[1], r[-2]) for r in routes]  # for single customer, both same
    # Build savings list
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                savings.append((s, i, j))
    savings.sort(reverse=True)
    
    # Merge until we have exactly truck_count routes
    # For fast lookup, map customer to route index
    cust_to_route = {c: idx for idx, r in enumerate(routes) for c in r[1:-1]}
    
    for saving, i, j in savings:
        if len(routes) <= truck_count:
            break
        if i not in cust_to_route or j not in cust_to_route:
            continue
        idx_i = cust_to_route[i]
        idx_j = cust_to_route[j]
        if idx_i == idx_j:
            continue
        route_i = routes[idx_i]
        route_j = routes[idx_j]
        # Check if i is at end of its route and j is at end of its route
        # i must be either first customer (index 1) or last customer (index -2) of route_i
        # same for j
        if not ((route_i[1] == i or route_i[-2] == i) and (route_j[1] == j or route_j[-2] == j)):
            continue
        # Determine orientation to connect i and j directly
        # We need to concatenate such that i and j become adjacent
        if route_i[1] == i and route_j[-2] == j:
            # i at start of route_i, j at end of route_j -> reverse order: route_j + route_i (but need to connect j to i)
            new_route = route_j[:-1] + route_i[1:]
        elif route_i[-2] == i and route_j[1] == j:
            new_route = route_i[:-1] + route_j[1:]
        elif route_i[1] == i and route_j[1] == j:
            # i at start, j at start -> reverse route_i and connect to route_j? This is tricky; standard savings only merges if one is end and other is start appropriately, but here both at start means we can't directly connect without reversing one route.
            # To keep it simple, we can reverse route_i so that i becomes last, then connect.
            # But reversing changes order; we need to ensure correctness. Let's allow reversal.
            reversed_i = [0] + route_i[1:-1][::-1] + [0]
            new_route = reversed_i[:-1] + route_j[1:]  # now i is last of reversed_i and j first of route_j
        elif route_i[-2] == i and route_j[-2] == j:
            reversed_j = [0] + route_j[1:-1][::-1] + [0]
            new_route = route_i[:-1] + reversed_j[1:]
        else:
            continue
        # Update cust_to_route for all customers in new_route
        for c in new_route[1:-1]:
            cust_to_route[c] = idx_i  # reuse idx_i
        # Remove route_j
        # Remove from cust_to_route? Actually we already updated for those customers to idx_i, but we need to delete old entries for customers in route_j? They are overwritten.
        # Remove route_j from routes list
        # We need to handle index shifting. Simpler: replace route_i with new_route, and delete route_j.
        # But deleting changes indices. We'll mark route_j as None and later filter.
        routes[idx_i] = new_route
        routes[idx_j] = None
        # After loop, filter out None
        routes = [r for r in routes if r is not None]
        # Rebuild cust_to_route? Actually after filtering, indices change, but we already updated for new_route. For customers in route_j, they are now part of route_i, so fine. But we need to reindex cust_to_route because route indices changed. Simpler: rebuild cust_to_route from scratch after each merge? But that's O(n^2). For n=100, fine.
        cust_to_route.clear()
        for idx, r in enumerate(routes):
            for c in r[1:-1]:
                cust_to_route[c] = idx
    
    # Possibly more routes than truck_count due to skipped savings; just take first truck_count routes (shouldn't happen)
    if len(routes) > truck_count:
        # Discard extra customers in extra routes? Not allowed; we must have exactly all customers. So we need to merge arbitrarily. This is a fallback: just merge the remaining routes somehow.
        # Simple: while len(routes) > truck_count, take two longest routes and merge them by concatenation.
        while len(routes) > truck_count:
            # find two routes with longest total distance?
            dists = [route_distance(r) for r in routes]
            # merge the two with longest max? Choose the two with largest total distance?
            # For simplicity, merge the last two routes.
            r1 = routes.pop()
            r2 = routes.pop()
            # Concatenate: r1 + r2[1:] to avoid duplicate depot
            new_r = r1[:-1] + r2[1:]
            routes.append(new_r)
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    report_best(routes)
    
    # Simulated Annealing
    max_iter = min(5000, n * truck_count * 10)  # bounded
    T_start = 10.0
    T_end = 0.001
    T = T_start
    current_routes = [list(r) for r in routes]
    current_max = max(route_distance(r) for r in current_routes)
    
    for it in range(max_iter):
        T = T_start * (T_end / T_start) ** (it / max_iter)
        # Select move type randomly
        move_type = random.randint(0, 2)
        routes_copy = [list(r) for r in current_routes]
        improved = False
        if move_type == 0:  # inter-route relocate
            # pick a random route that has interior customers
            nonempty = [idx for idx, r in enumerate(routes_copy) if len(r) > 2]
            if not nonempty:
                continue
            src_idx = random.choice(nonempty)
            src_route = routes_copy[src_idx]
            # pick a random customer from interior
            interior = src_route[1:-1]
            cust = random.choice(interior)
            src_route.remove(cust)
            # pick a random destination route and position
            dest_idx = random.randint(0, truck_count-1)
            while dest_idx == src_idx and len(routes_copy[dest_idx]) == 0:
                # avoid inserting into empty route? Actually empty route is [0,0] which has len 2, but interior empty; we can insert between depots.
                # We can still insert at position 1.
                dest_idx = random.randint(0, truck_count-1)
            dest_route = routes_copy[dest_idx]
            # choose random position between 1 and len(dest_route)-1 inclusive
            pos = random.randint(1, len(dest_route)-1)
            dest_route.insert(pos, cust)
        elif move_type == 1:  # inter-route swap
            # pick two different routes with interior customers
            nonempty = [idx for idx, r in enumerate(routes_copy) if len(r) > 2]
            if len(nonempty) < 2:
                continue
            idx1, idx2 = random.sample(nonempty, 2)
            route1 = routes_copy[idx1]
            route2 = routes_copy[idx2]
            # pick random interior customers
            cust1 = random.choice(route1[1:-1])
            cust2 = random.choice(route2[1:-1])
            # swap them
            i1 = route1.index(cust1)
            i2 = route2.index(cust2)
            route1[i1] = cust2
            route2[i2] = cust1
        else:  # intra-route 2-opt
            # pick a random route with at least 4 nodes (including depot) to have a segment
            long = [idx for idx, r in enumerate(routes_copy) if len(r) > 3]
            if not long:
                continue
            idx = random.choice(long)
            route = routes_copy[idx]
            # pick random start and end indices for reversal (excluding depots)
            if len(route) <= 3:
                continue
            start = random.randint(1, len(route)-3)
            end = random.randint(start+1, len(route)-2)
            route[start:end+1] = reversed(route[start:end+1])
        
        # evaluate new max
        new_max = max(route_distance(r) for r in routes_copy)
        delta = new_max - current_max
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_routes = routes_copy
            current_max = new_max
            if new_max < best_max - 1e-12:
                report_best(current_routes)
    
    return best_routes if best_routes is not None else routes
import numpy as np
from typing import List

def report_best_vrp(routes):
    pass

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    n = distance_matrix.shape[0]
    # Initialize each customer in its own route
    routes = [[0, i, 0] for i in range(1, n)]
    # Compute savings for all pairs of customers
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
            savings.append((s, i, j))
    # Sort descending by savings, then by i, then j
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    # Map customer to route index
    customer_to_route = {i: idx for idx, route in enumerate(routes) for i in route[1:-1]}
    # Helper to get endpoints of a route (first and last customer excluding depot)
    def route_endpoints(route):
        return route[1], route[-2]
    # Merge process
    for s_val, i, j in savings:
        if len(routes) <= truck_count:
            break
        if i not in customer_to_route or j not in customer_to_route:
            continue
        ri = customer_to_route[i]
        rj = customer_to_route[j]
        if ri == rj:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        # Check if merge is possible: endpoints must be i and j in correct order
        first_i, last_i = route_endpoints(route_i)
        first_j, last_j = route_endpoints(route_j)
        # Check both orientations
        if (last_i == i and first_j == j):
            # Merge: route_i + route_j[1:]
            new_route = route_i + route_j[1:]
        elif (last_j == j and first_i == i):
            new_route = route_j + route_i[1:]
        else:
            continue
        # Update customer_to_route for customers in route_j
        for cust in route_j[1:-1]:
            customer_to_route[cust] = ri
        # Remove route_j
        routes.pop(rj)
        routes[ri] = new_route
        # Update customer_to_route index for later routes (shift)
        for k in range(rj, len(routes)):
            for cust in routes[k][1:-1]:
                customer_to_route[cust] = k
    # If still too many routes, force merge with smallest penalty
    # Already exhausted savings? If len(routes) > truck_count, we need to continue merging any remaining pairs
    # We'll iterate over all customer pairs again, ignoring savings order but with feasible merge
    if len(routes) > truck_count:
        # collect all pairs again
        all_pairs = [(i, j) for i in range(1, n) for j in range(i+1, n)]
        for i, j in all_pairs:
            if len(routes) <= truck_count:
                break
            if i not in customer_to_route or j not in customer_to_route:
                continue
            ri = customer_to_route[i]
            rj = customer_to_route[j]
            if ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            first_i, last_i = route_endpoints(route_i)
            first_j, last_j = route_endpoints(route_j)
            if (last_i == i and first_j == j) or (last_j == j and first_i == i):
                if (last_i == i and first_j == j):
                    new_route = route_i + route_j[1:]
                else:
                    new_route = route_j + route_i[1:]
                for cust in route_j[1:-1]:
                    customer_to_route[cust] = ri
                routes.pop(rj)
                routes[ri] = new_route
                for k in range(rj, len(routes)):
                    for cust in routes[k][1:-1]:
                        customer_to_route[cust] = k
    # Add empty routes if needed
    while len(routes) < truck_count:
        routes.append([0, 0])
    # Compute route distances
    def route_distance(route):
        d = 0
        for p in range(len(route)-1):
            d += distance_matrix[route[p], route[p+1]]
        return d
    report_best_vrp(routes)
    # Improvement: rebalance to minimize max distance
    max_iter = 10
    for _ in range(max_iter):
        # Compute distances
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda k: dists[k])
        min_idx = min(range(len(dists)), key=lambda k: dists[k])
        best_improvement = 0
        best_move = None
        route_max = routes[max_idx]
        # Try moving each customer from max route to min route (or any other)
        for pos in range(1, len(route_max)-1):
            cust = route_max[pos]
            for target_idx in range(len(routes)):
                if target_idx == max_idx:
                    continue
                target_route = routes[target_idx]
                # Try inserting cust in all positions
                for ins in range(1, len(target_route)):
                    new_max_route = route_max[:pos] + route_max[pos+1:]
                    new_max_dist = route_distance(new_max_route)
                    new_target = target_route[:ins] + [cust] + target_route[ins:]
                    new_target_dist = route_distance(new_target)
                    # Check if max distance decreases
                    new_dists = dists[:]
                    new_dists[max_idx] = new_max_dist
                    new_dists[target_idx] = new_target_dist
                    new_max = max(new_dists)
                    if new_max < max(dists):
                        improvement = max(dists) - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (max_idx, pos, target_idx, ins)
        if best_move:
            max_idx, pos, target_idx, ins = best_move
            cust = routes[max_idx][pos]
            new_max_route = routes[max_idx][:pos] + routes[max_idx][pos+1:]
            if len(new_max_route) == 2:
                routes[max_idx] = [0, 0]
            else:
                routes[max_idx] = new_max_route
            routes[target_idx] = routes[target_idx][:ins] + [cust] + routes[target_idx][ins:]
            report_best_vrp(routes)
        else:
            break
    report_best_vrp(routes)
    return routes
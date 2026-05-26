import numpy as np
from collections import defaultdict

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n - 1 == 0:
        return [[0,0] for _ in range(truck_count)]
    # Initialization: each customer as a separate route
    routes = [[0, i, 0] for i in customers]
    if len(routes) <= truck_count:
        while len(routes) < truck_count:
            routes.append([0,0])
        # Still need to rebalance? For now just report and return
        report_best_vrp(routes)
        return routes
    # Build savings list
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0,i] + distance_matrix[0,j] - distance_matrix[i,j]
                savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])
    # Map customer to route index
    cust_to_route = {i: idx for idx, r in enumerate(routes) for i in r[1:-1]}
    # Track first and last customer of each route
    route_first = {idx: r[1] for idx, r in enumerate(routes)}
    route_last = {idx: r[-2] for idx, r in enumerate(routes)}
    # Merge until truck_count routes
    for s, i, j in savings:
        if len(routes) == truck_count:
            break
        ri = cust_to_route.get(i)
        rj = cust_to_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        # Check if i is last of its route and j is first of its route
        if route_last[ri] == i and route_first[rj] == j:
            new_route = routes[ri][:-1] + routes[rj][1:]
        elif route_first[ri] == i and route_last[rj] == j:
            new_route = routes[rj][:-1] + routes[ri][1:]
        else:
            continue
        # Remove old routes (higher index first to avoid shifting issues)
        ri, rj = sorted([ri, rj], reverse=True)
        routes.pop(ri)
        routes.pop(rj)
        # Add new route
        routes.append(new_route)
        new_idx = len(routes) - 1
        # Update mappings
        for cust in new_route[1:-1]:
            cust_to_route[cust] = new_idx
        route_first[new_idx] = new_route[1]
        route_last[new_idx] = new_route[-2]
    # Add empty routes if fewer than truck_count
    while len(routes) < truck_count:
        routes.append([0,0])
    # Report initial solution
    report_best_vrp(routes)
    # Rebalancing improvement
    max_iter = (n-1) * truck_count
    for _ in range(max_iter):
        dists = [route_distance(r, distance_matrix) for r in routes]
        max_dist = max(dists)
        max_idx = dists.index(max_dist)
        best_improvement = 0.0
        best_move = None
        longest_route = routes[max_idx]
        if len(longest_route) <= 2:
            break
        for pos_cust in range(1, len(longest_route)-1):
            customer = longest_route[pos_cust]
            for other_idx, other_route in enumerate(routes):
                if other_idx == max_idx:
                    continue
                for insert_pos in range(1, len(other_route)):
                    # Create new routes
                    new_long = longest_route[:pos_cust] + longest_route[pos_cust+1:]
                    new_other = other_route[:insert_pos] + [customer] + other_route[insert_pos:]
                    new_dists = dists.copy()
                    new_dists[max_idx] = route_distance(new_long, distance_matrix)
                    new_dists[other_idx] = route_distance(new_other, distance_matrix)
                    new_max = max(new_dists)
                    if new_max < max_dist - 1e-12:
                        improvement = max_dist - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (max_idx, pos_cust, other_idx, insert_pos)
        if best_move is not None:
            max_idx, pos_cust, other_idx, insert_pos = best_move
            customer = routes[max_idx][pos_cust]
            routes[max_idx] = routes[max_idx][:pos_cust] + routes[max_idx][pos_cust+1:]
            routes[other_idx] = routes[other_idx][:insert_pos] + [customer] + routes[other_idx][insert_pos:]
            report_best_vrp(routes)
        else:
            break
    report_best_vrp(routes)
    return routes
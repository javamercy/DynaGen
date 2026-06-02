import numpy as np
from typing import List

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    # Initialize routes: each customer alone
    routes = [[0, i, 0] for i in range(1, n)]
    # If no customers, return empty routes
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    # Compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((s, i, j))
    # Sort by savings descending, then i, then j
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    # Track which route each customer belongs to and the endpoints
    customer_route = {i: i-1 for i in range(1, n)}  # route index
    route_set = {idx: set(route[1:-1]) for idx, route in enumerate(routes)}
    # For each route, we need to know if customer i is at the end? Actually we maintain route first and last customer
    route_first = {i-1: i for i in range(1, n)}  # route index -> first customer after depot
    route_last = {i-1: i for i in range(1, n)}   # route index -> last customer before depot
    # We'll merge until we have truck_count routes or no more savings
    while len(routes) > truck_count:
        merged = False
        for _, i, j in savings:
            if len(routes) <= truck_count:
                break
            ri = customer_route.get(i)
            rj = customer_route.get(j)
            if ri is None or rj is None:
                continue
            if ri == rj:
                continue
            # Check if i is at end of its route and j is at start, or vice versa
            route_i = routes[ri]
            route_j = routes[rj]
            # i at end: route_i[-2] == i (since last is depot)
            i_at_end = (route_i[-2] == i)
            i_at_start = (route_i[1] == i)
            j_at_end = (route_j[-2] == j)
            j_at_start = (route_j[1] == j)
            feasible = False
            if i_at_end and j_at_start:
                # merge: route_i then route_j (skip depot in between)
                new_route = route_i[:-1] + route_j[1:]
                feasible = True
            elif i_at_start and j_at_end:
                # merge: route_j then route_i
                new_route = route_j[:-1] + route_i[1:]
                feasible = True
            if feasible:
                # Remove old routes
                routes.pop(max(ri, rj))
                routes.pop(min(ri, rj))
                # Add new route
                routes.append(new_route)
                # Update customer_route and route_set
                new_idx = len(routes) - 1
                for cust in new_route[1:-1]:
                    customer_route[cust] = new_idx
                # Update route_first and route_last (not really needed after merge)
                # Remove old route indices from structure
                merged = True
                break
        if not merged:
            break
    # If we have more routes than truck_count, we need to merge arbitrarily (but should not happen if savings allow)
    # Just merge any two routes until truck_count
    while len(routes) > truck_count:
        # merge first two routes
        r1 = routes.pop(0)
        r2 = routes.pop(0)
        new_route = r1[:-1] + r2[1:]
        routes.append(new_route)
    # Add empty routes if less than truck_count
    while len(routes) < truck_count:
        routes.append([0, 0])
    # Ensure each route starts and ends with 0
    for route in routes:
        if route[0] != 0:
            route.insert(0, 0)
        if route[-1] != 0:
            route.append(0)
    # Call report_best_vrp (assumed available)
    try:
        from types import FunctionType
        # report_best_vrp is a built-in provided function
        report_best_vrp(routes)
    except NameError:
        pass
    return routes
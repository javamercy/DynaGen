import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        best = route[:]
        best_d = route_distance(route)
        max_iter = n * 2
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new)
                    if d < best_d - 1e-12:
                        best_d = d
                        best = new
                        improved = True
            route = best
            iter_count += 1
        return best

    # Clarke-Wright savings algorithm
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
            savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    # Initialize routes: each customer in its own route (return to depot)
    routes = {i: [0, i, 0] for i in range(1, n)}
    node_to_route = {i: i for i in range(1, n)}
    used = set()

    for s, i, j in savings:
        if i in used or j in used:
            continue
        ri = node_to_route[i]
        rj = node_to_route[j]
        if ri == rj:
            continue
        # Check if i and j are endpoints (adjacent to depot)
        ri_route = routes[ri]
        rj_route = routes[rj]
        if ri_route[1] == i and rj_route[-2] == j:
            # Combine: append rj after i
            new_route = ri_route[:-1] + rj_route[1:]
            routes[ri] = new_route
            routes.pop(rj)
            node_to_route[j] = ri
            for node in rj_route[1:-1]:
                node_to_route[node] = ri
            used.add(i)
            used.add(j)
        elif ri_route[-2] == i and rj_route[1] == j:
            # Combine: append ri before rj
            new_route = ri_route[:-1] + rj_route[1:]
            routes[ri] = new_route
            routes.pop(rj)
            node_to_route[j] = ri
            for node in rj_route[1:-1]:
                node_to_route[node] = ri
            used.add(i)
            used.add(j)
        # else skip

    # Convert to list of routes
    current_routes = list(routes.values())
    # Ensure exactly truck_count routes: if too many, merge smallest routes into largest?
    # Instead, we'll create extra empty routes if needed later.

    # Post-process: ensure no duplicate customers
    all_customers = set()
    for r in current_routes:
        for c in r[1:-1]:
            all_customers.add(c)
    # merge missing customers into route with largest savings?
    missing = [c for c in range(1, n) if c not in all_customers]
    for c in missing:
        # insert into shortest route? assign to route with closest depot? simple: append to first route
        current_routes[0].insert(-1, c)

    # Apply 2-opt to all routes
    for t in range(len(current_routes)):
        current_routes[t] = two_opt(current_routes[t])

    # If more routes than truck_count, merge the smallest into others?
    # But we need exactly truck_count. If len(current_routes) > truck_count, merge smallest routes into longest.
    while len(current_routes) > truck_count:
        # find the route with smallest number of customers (excluding depot)
        sizes = [len(r)-2 for r in current_routes]
        min_idx = min(range(len(current_routes)), key=lambda i: (sizes[i], total_distance([current_routes[i]])))
        min_route = current_routes.pop(min_idx)
        # merge its customers into the route with smallest increase in max distance?
        # simple: insert into longest route
        max_idx = max(range(len(current_routes)), key=lambda i: route_distance(current_routes[i]))
        # insert customers one by one at best position (min distance increase)
        for cust in min_route[1:-1]:
            best_pos = 1
            best_inc = float('inf')
            for pos in range(1, len(current_routes[max_idx])):
                new_route = current_routes[max_idx][:pos] + [cust] + current_routes[max_idx][pos:]
                inc = route_distance(new_route) - route_distance(current_routes[max_idx])
                if inc < best_inc - 1e-12:
                    best_inc = inc
                    best_pos = pos
            current_routes[max_idx] = current_routes[max_idx][:best_pos] + [cust] + current_routes[max_idx][best_pos:]
        # apply two-opt to modified route
        current_routes[max_idx] = two_opt(current_routes[max_idx])

    # If fewer routes than truck_count, add empty routes
    while len(current_routes) < truck_count:
        current_routes.append([0, 0])

    # Improvement: simple relocation from max route to others to reduce max distance
    best_routes = [r[:] for r in current_routes]
    best_max = max_distance(best_routes)
    report_best_vrp(best_routes)

    improvement = True
    max_iter = n * 2
    iter_count = 0
    while improvement and iter_count < max_iter:
        improvement = False
        # Find route with max distance
        max_idx = max(range(truck_count), key=lambda t: route_distance(current_routes[t]))
        max_route = current_routes[max_idx]
        if len(max_route) <= 2:
            break
        # Try relocating each customer in max_route to other routes
        best_improvement = None
        best_new_max = float('inf')
        best_new_total = float('inf')
        for idx in range(1, len(max_route)-1):
            cust = max_route[idx]
            new_max_route = max_route[:idx] + max_route[idx+1:]
            for t2 in range(truck_count):
                if t2 == max_idx:
                    continue
                r2 = current_routes[t2]
                for pos in range(1, len(r2)):
                    new_r2 = r2[:pos] + [cust] + r2[pos:]
                    d_max_new = route_distance(new_max_route)
                    d2_new = route_distance(new_r2)
                    other_max = 0.0
                    other_total = 0.0
                    for idx2, r in enumerate(current_routes):
                        if idx2 not in (max_idx, t2):
                            d = route_distance(r)
                            if d > other_max:
                                other_max = d
                            other_total += d
                    cand_max = max(d_max_new, d2_new, other_max)
                    cand_total = d_max_new + d2_new + other_total
                    if cand_max < best_new_max - 1e-12 or (abs(cand_max - best_new_max) < 1e-12 and cand_total < best_new_total - 1e-12):
                        best_new_max = cand_max
                        best_new_total = cand_total
                        best_improvement = (t2, pos, new_max_route, new_r2)
        if best_improvement is not None and best_new_max < best_max - 1e-12:
            t2, pos, new_max_route, new_r2 = best_improvement
            current_routes[max_idx] = two_opt(new_max_route)
            current_routes[t2] = two_opt(new_r2)
            best_max = max_distance(current_routes)
            best_routes = [r[:] for r in current_routes]
            report_best_vrp(best_routes)
            improvement = True
            iter_count += 1

    # Final 2-opt
    for t in range(truck_count):
        best_routes[t] = two_opt(best_routes[t])
    best_max = max_distance(best_routes)
    report_best_vrp(best_routes)

    return best_routes
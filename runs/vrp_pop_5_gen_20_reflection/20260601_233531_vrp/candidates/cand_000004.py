import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    random.seed(0)

    # Initial solution: assign each customer to its own dummy route? No, we need exactly truck_count routes.
    # Use farthest-first insertion: sort customers by distance to depot descending.
    depot_dist = [distance_matrix[0][c] for c in customers]
    sorted_customers = [c for _, c in sorted(zip(depot_dist, customers), reverse=True)]

    # Initialize routes: all empty [0,0]
    routes = [[0, 0] for _ in range(truck_count)]

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        return dist

    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)

    # Insert customers one by one
    for cust in sorted_customers:
        best_increase = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            # consider all insertion positions (after 0 and before 0 at end, but route ends with 0)
            for pos in range(1, len(route)):
                # compute increase in distance for this route
                prev_dist = distance_matrix[route[pos-1]][route[pos]]
                new_dist = distance_matrix[route[pos-1]][cust] + distance_matrix[cust][route[pos]]
                increase = new_dist - prev_dist
                if increase < best_increase:
                    best_increase = increase
                    best_route_idx = r_idx
                    best_pos = pos
        # Insert at best position
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        # Call report_best_vrp if improved? We'll call after full construction.

    # Report initial solution
    from importlib import import_module
    report_best_vrp = import_module(__name__).__dict__.get('report_best_vrp', lambda x: None)
    # Try to get report_best_vrp from the executing module; if not found, use lambda.
    # Actually, we need to call report_best_vrp if it exists. Assume it's defined in the environment.
    # We'll use try-except.
    try:
        report_best_vrp(routes)
    except:
        pass

    # Local search: iterate until no improvement or max iterations
    max_iter = n * n  # finite bound
    for iteration in range(max_iter):
        improved = False
        current_max = max_route_distance(routes)
        best_routes = [route[:] for route in routes]
        best_max = current_max

        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    # check if this reduces max route distance
                    # compute new max
                    old_dist_i = route_distance(route)
                    new_dist_i = route_distance(new_route)
                    # find other routes distances
                    other_dists = [route_distance(routes[k]) for k in range(truck_count) if k != r_idx]
                    old_max = max(old_dist_i, max(other_dists) if other_dists else 0)
                    new_max = max(new_dist_i, max(other_dists) if other_dists else 0)
                    if new_max < old_max - 1e-9:
                        # improvement found, apply immediately (first improvement)
                        routes[r_idx] = new_route
                        improved = True
                        # recalc current_max
                        current_max = max(route_distance(new_route), max(other_dists) if other_dists else 0)
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            try:
                report_best_vrp(routes)
            except:
                pass
            continue

        # Inter-route relocate: move a customer from one route to another
        for r_idx_from in range(truck_count):
            route_from = routes[r_idx_from]
            if len(route_from) <= 2:
                continue
            for cust in route_from[1:-1]:  # skip depot
                # try moving to other routes
                for r_idx_to in range(truck_count):
                    if r_idx_to == r_idx_from:
                        continue
                    route_to = routes[r_idx_to]
                    # try all insertion positions
                    for pos in range(1, len(route_to)):
                        # compute new distances
                        old_dist_from = route_distance(route_from)
                        old_dist_to = route_distance(route_to)
                        # remove cust from route_from
                        new_route_from = [x for x in route_from if x != cust]
                        new_route_to = route_to[:pos] + [cust] + route_to[pos:]
                        new_dist_from = route_distance(new_route_from)
                        new_dist_to = route_distance(new_route_to)
                        # compute max
                        other_dists = [route_distance(routes[k]) for k in range(truck_count) if k not in (r_idx_from, r_idx_to)]
                        old_max = max(old_dist_from, old_dist_to, max(other_dists) if other_dists else 0)
                        new_max = max(new_dist_from, new_dist_to, max(other_dists) if other_dists else 0)
                        if new_max < old_max - 1e-9:
                            routes[r_idx_from] = new_route_from
                            routes[r_idx_to] = new_route_to
                            improved = True
                            try:
                                report_best_vrp(routes)
                            except:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Inter-route exchange (swap two customers)
        for r_idx1 in range(truck_count):
            route1 = routes[r_idx1]
            if len(route1) <= 2:
                continue
            for i in range(1, len(route1)-1):
                cust1 = route1[i]
                for r_idx2 in range(r_idx1+1, truck_count):
                    route2 = routes[r_idx2]
                    if len(route2) <= 2:
                        continue
                    for j in range(1, len(route2)-1):
                        cust2 = route2[j]
                        # swap
                        new_route1 = route1[:i] + [cust2] + route1[i+1:]
                        new_route2 = route2[:j] + [cust1] + route2[j+1:]
                        old_dist1 = route_distance(route1)
                        old_dist2 = route_distance(route2)
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_dists = [route_distance(routes[k]) for k in range(truck_count) if k not in (r_idx1, r_idx2)]
                        old_max = max(old_dist1, old_dist2, max(other_dists) if other_dists else 0)
                        new_max = max(new_dist1, new_dist2, max(other_dists) if other_dists else 0)
                        if new_max < old_max - 1e-9:
                            routes[r_idx1] = new_route1
                            routes[r_idx2] = new_route2
                            improved = True
                            try:
                                report_best_vrp(routes)
                            except:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    # Ensure all customers are assigned exactly once (should be)
    visited = [False]*n
    for route in routes:
        for node in route[1:-1]:
            visited[node] = True
    assert all(visited[1:]), "some customers missing"

    return routes
import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def insert_cost(route, node):
        best_cost = float('inf')
        best_pos = -1
        for i in range(1, len(route)):
            cost = distance_matrix[route[i-1], node] + distance_matrix[node, route[i]] - distance_matrix[route[i-1], route[i]]
            if cost < best_cost:
                best_cost = cost
                best_pos = i
        return best_cost, best_pos

    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    current_dist = [0.0 for _ in range(truck_count)]
    customers = list(range(1, n))

    # Construct initial solution (same as parent)
    for cust in customers:
        best_route = -1
        best_new_max = float('inf')
        best_pos = -1
        for r in range(truck_count):
            cost, pos = insert_cost(routes[r], cust)
            new_dist = current_dist[r] + cost
            old_max = max(current_dist)
            if r == current_dist.index(old_max):
                other_dists = [current_dist[i] for i in range(truck_count) if i != r]
                new_max = max(new_dist, *other_dists)
            else:
                new_max = max(new_dist, old_max)
            if new_max < best_new_max or (new_max == best_new_max and new_dist < current_dist[best_route]):
                best_new_max = new_max
                best_route = r
                best_pos = pos
        routes[best_route].insert(best_pos, cust)
        current_dist[best_route] = route_distance(routes[best_route])

    best_routes = [list(r) for r in routes]
    best_max = max(current_dist)
    report_best_vrp(routes)

    def local_search(best_routes, best_max, current_dist, routes):
        # Best-improvement local search on each neighborhood, repeatedly until no improvement
        n_cust = n - 1
        max_iters = 10 * n_cust * truck_count
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Best relocate
            best_move = None
            best_new_max = float('inf')
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for idx in range(1, len(route1)-1):
                    cust = route1[idx]
                    new_route1 = route1[:idx] + route1[idx+1:]
                    new_dist1 = route_distance(new_route1)
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        cost, pos = insert_cost(route2, cust)
                        new_route2 = route2[:pos] + [cust] + route2[pos:]
                        new_dist2 = current_dist[r2] + cost
                        other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1,r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = (r1, idx, r2, pos, new_route1, new_route2, new_dist1, new_dist2)
            if best_move is not None and best_new_max < best_max:
                r1, idx, r2, pos, new_route1, new_route2, new_dist1, new_dist2 = best_move
                routes[r1] = new_route1
                routes[r2] = new_route2
                current_dist[r1] = new_dist1
                current_dist[r2] = new_dist2
                best_max = best_new_max
                best_routes = [list(r) for r in routes]
                improved = True
                report_best_vrp(routes)
                continue

            # Best swap
            best_move = None
            best_new_max = float('inf')
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for idx1 in range(1, len(route1)-1):
                    cust1 = route1[idx1]
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 2:
                            continue
                        for idx2 in range(1, len(route2)-1):
                            cust2 = route2[idx2]
                            new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                            new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                            new_dist1 = route_distance(new_route1)
                            new_dist2 = route_distance(new_route2)
                            other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1,r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = (r1, idx1, r2, idx2, new_route1, new_route2, new_dist1, new_dist2)
            if best_move is not None and best_new_max < best_max:
                r1, idx1, r2, idx2, new_route1, new_route2, new_dist1, new_dist2 = best_move
                routes[r1] = new_route1
                routes[r2] = new_route2
                current_dist[r1] = new_dist1
                current_dist[r2] = new_dist2
                best_max = best_new_max
                best_routes = [list(r) for r in routes]
                improved = True
                report_best_vrp(routes)
                continue

            # Best intra 2-opt
            best_move = None
            best_improv = 0
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < current_dist[r] - 1e-9:
                            improv = current_dist[r] - new_dist
                            if improv > best_improv:
                                best_improv = improv
                                best_move = (r, i, j, new_route, new_dist)
            if best_move is not None:
                r, i, j, new_route, new_dist = best_move
                routes[r] = new_route
                current_dist[r] = new_dist
                new_max = max(current_dist)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(routes)
                improved = True
                continue

            # Best cross 2-opt
            best_move = None
            best_new_max = float('inf')
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i+1] + route2[j+1:]
                            new2 = route2[:j+1] + route1[i+1:]
                            new_dist1 = route_distance(new1)
                            new_dist2 = route_distance(new2)
                            other_dists = [current_dist[k] for k in range(truck_count) if k not in (r1,r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = (r1, r2, new1, new2, new_dist1, new_dist2)
            if best_move is not None and best_new_max < best_max:
                r1, r2, new1, new2, new_dist1, new_dist2 = best_move
                routes[r1] = new1
                routes[r2] = new2
                current_dist[r1] = new_dist1
                current_dist[r2] = new_dist2
                best_max = best_new_max
                best_routes = [list(r) for r in routes]
                improved = True
                report_best_vrp(routes)
                continue

        return best_routes, best_max, current_dist, routes

    # First local search pass
    best_routes, best_max, current_dist, routes = local_search(best_routes, best_max, current_dist, routes)

    # ILS: perturb and re-optimize with adaptive iterations
    max_ils = 10
    no_improve_count = 0
    for _ in range(max_ils):
        # Perturb: move two customers from longest route to shortest route
        # Find longest route index
        max_dist = max(current_dist)
        long_routes = [r for r, d in enumerate(current_dist) if d == max_dist]
        if not long_routes:
            break
        r_long = long_routes[0]
        route_long = routes[r_long]
        if len(route_long) <= 3:  # at least one customer
            # if only one customer, move one; otherwise skip
            # move one customer if possible
            if len(route_long) <= 2:
                break
            # move first customer
            cust_idx = 1
            cust = route_long[cust_idx]
            new_route_long = route_long[:cust_idx] + route_long[cust_idx+1:]
            new_dist_long = route_distance(new_route_long)
            # find shortest route
            min_dist = min(current_dist)
            short_routes = [r for r, d in enumerate(current_dist) if d == min_dist]
            r_short = short_routes[0]
            route_short = routes[r_short]
            cost, pos = insert_cost(route_short, cust)
            new_route_short = route_short[:pos] + [cust] + route_short[pos:]
            new_dist_short = current_dist[r_short] + cost
            # apply perturbation
            routes[r_long] = new_route_long
            routes[r_short] = new_route_short
            current_dist[r_long] = new_dist_long
            current_dist[r_short] = new_dist_short
        else:
            # move two customers sequentially: first and then next (after removal, becomes first again)
            # First move
            cust_idx1 = 1
            cust1 = route_long[cust_idx1]
            new_route_long1 = route_long[:cust_idx1] + route_long[cust_idx1+1:]
            new_dist_long1 = route_distance(new_route_long1)
            # find shortest route
            min_dist = min(current_dist)
            short_routes = [r for r, d in enumerate(current_dist) if d == min_dist]
            r_short = short_routes[0]
            route_short = routes[r_short]
            cost1, pos1 = insert_cost(route_short, cust1)
            new_route_short1 = route_short[:pos1] + [cust1] + route_short[pos1:]
            new_dist_short1 = current_dist[r_short] + cost1
            # apply first move
            routes[r_long] = new_route_long1
            routes[r_short] = new_route_short1
            current_dist[r_long] = new_dist_long1
            current_dist[r_short] = new_dist_short1
            # Second move: now the longest route (currently r_long) has one less customer, take first again
            route_long2 = routes[r_long]
            if len(route_long2) > 2:  # still has customers
                cust_idx2 = 1
                cust2 = route_long2[cust_idx2]
                new_route_long2 = route_long2[:cust_idx2] + route_long2[cust_idx2+1:]
                new_dist_long2 = route_distance(new_route_long2)
                # again find shortest (current shortest may have changed)
                min_dist2 = min(current_dist)
                short_routes2 = [r for r, d in enumerate(current_dist) if d == min_dist2]
                r_short2 = short_routes2[0]
                route_short2 = routes[r_short2]
                cost2, pos2 = insert_cost(route_short2, cust2)
                new_route_short2 = route_short2[:pos2] + [cust2] + route_short2[pos2:]
                new_dist_short2 = current_dist[r_short2] + cost2
                # apply second move
                routes[r_long] = new_route_long2
                routes[r_short2] = new_route_short2
                current_dist[r_long] = new_dist_long2
                current_dist[r_short2] = new_dist_short2

        # Re-run local search
        old_best_max = best_max
        best_routes, best_max, current_dist, routes = local_search(best_routes, best_max, current_dist, routes)
        if best_max < old_best_max - 1e-9:
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= 2:
            break

    return best_routes
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

    def construct(insertion_order):
        routes = [[0, 0] for _ in range(truck_count)]
        current_dist = [0.0 for _ in range(truck_count)]
        for cust in insertion_order:
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
        return routes, current_dist

    # Initial sequential insertion
    customers_seq = list(range(1, n))
    routes, current_dist = construct(customers_seq)
    best_routes = [list(r) for r in routes]
    best_max = max(current_dist)

    def local_search(routes, current_dist, best_routes, best_max):
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
                continue

        return best_routes, best_max, current_dist, routes

    # First local search
    best_routes, best_max, current_dist, routes = local_search(routes, current_dist, best_routes, best_max)

    # ILS with block perturbation
    num_ils = 5
    no_improve_count = 0
    for _ in range(num_ils):
        # Perturb: move a block of 2 customers from longest route to shortest route
        # Find longest route
        max_dist = max(current_dist)
        long_routes = [r for r, d in enumerate(current_dist) if d == max_dist]
        if not long_routes:
            break
        r_long = long_routes[0]
        route_long = routes[r_long]
        if len(route_long) <= 3:  # need at least 3 nodes to have a block of 2 customers
            # Try moving one customer instead
            if len(route_long) <= 2:
                break
            block_size = 1
        else:
            block_size = 2
        # Determine starting index for block (first customer index 1)
        start_idx = 1
        # Ensure block fits
        if start_idx + block_size > len(route_long) - 1:
            start_idx = len(route_long) - 1 - block_size
        # Extract block
        block = route_long[start_idx:start_idx+block_size]
        # Remove block from long route
        new_route_long = route_long[:start_idx] + route_long[start_idx+block_size:]
        new_dist_long = route_distance(new_route_long)
        # Find shortest route
        min_dist = min(current_dist)
        short_routes = [r for r, d in enumerate(current_dist) if d == min_dist]
        r_short = short_routes[0]
        route_short = routes[r_short]
        # Insert block as a contiguous block? For simplicity, insert each customer individually in the best position (greedy)
        # But to keep perturbation strong, we can insert them as a block in the best insertion position for the block
        # Compute best position to insert block
        if len(block) == 0:
            break
        # Just insert each customer sequentially at best position (like first see effect)
        temp_route = list(route_short)
        temp_dist = current_dist[r_short]
        for cust in block:
            cost, pos = insert_cost(temp_route, cust)
            temp_route = temp_route[:pos] + [cust] + temp_route[pos:]
            temp_dist += cost
        new_route_short = temp_route
        new_dist_short = temp_dist
        # Apply perturbation
        routes[r_long] = new_route_long
        routes[r_short] = new_route_short
        current_dist[r_long] = new_dist_long
        current_dist[r_short] = new_dist_short
        # Re-run local search
        best_routes, best_max, current_dist, routes = local_search(routes, current_dist, best_routes, best_max)
        # Check for improvement
        current_max = max(current_dist)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            no_improve_count = 0
        else:
            no_improve_count += 1
        # If no improvement in 3 ILS iterations, restart with farthest-first
        if no_improve_count >= 3:
            # Construct farthest-first insertion order
            # start from depot 0, then farthest customer, then farthest from current set?
            # Simple: sort customers by distance from depot descending
            dist_from_depot = [distance_matrix[0][c] for c in range(1, n)]
            order = sorted(range(1, n), key=lambda x: dist_from_depot[x-1], reverse=True)
            routes, current_dist = construct(order)
            best_routes, best_max, current_dist, routes = local_search(routes, current_dist, best_routes, best_max)
            no_improve_count = 0

    return best_routes
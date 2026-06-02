import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    # Initialize routes: each route starts and ends at depot
    routes = [[depot, depot] for _ in range(truck_count)]

    # ---- Construction: Cheapest insertion to minimize max route distance ----
    # Sort customers by distance to depot (descending), then by index for tie-breaking
    sorted_customers = sorted(customers, key=lambda c: (-distance_matrix[depot][c], c))

    for cust in sorted_customers:
        best_max = math.inf
        best_route_idx = -1
        best_insert_pos = -1

        for r_idx in range(truck_count):
            route = routes[r_idx]
            # Evaluate insertion positions between depot and depot (indices 1 to len(route)-1)
            for pos in range(1, len(route)):
                # Compute new route distance after insertion
                old_dist = 0
                for i in range(len(route)-1):
                    old_dist += distance_matrix[route[i]][route[i+1]]
                new_dist = old_dist - distance_matrix[route[pos-1]][route[pos]] + distance_matrix[route[pos-1]][cust] + distance_matrix[cust][route[pos]]
                # Compute new max route distance if we update this route
                current_max = 0
                for r2_idx, r2 in enumerate(routes):
                    if r2_idx == r_idx:
                        d = new_dist
                    else:
                        d = 0
                        for i in range(len(r2)-1):
                            d += distance_matrix[r2[i]][r2[i+1]]
                    if d > current_max:
                        current_max = d
                if current_max < best_max:
                    best_max = current_max
                    best_route_idx = r_idx
                    best_insert_pos = pos

        # Insert into best route at best position
        route = routes[best_route_idx]
        route.insert(best_insert_pos, cust)

    # Report initial solution
    report_best_vrp(routes)

    # ---- Improvement: Local search (relocate only) ----
    max_iter = n  # bounded by instance size
    for iteration in range(max_iter):
        # Compute current route distances and max
        route_dists = []
        for r in routes:
            d = 0
            for i in range(len(r)-1):
                d += distance_matrix[r[i]][r[i+1]]
            route_dists.append(d)
        current_max = max(route_dists)
        # Find longest route (tie-break by index)
        longest_idx = min(i for i, d in enumerate(route_dists) if d == current_max)

        best_move = None
        best_new_max = current_max

        # Relocate moves: try to move a customer from the longest route to another route
        if len(routes[longest_idx]) > 2:  # at least one customer
            for cust_pos in range(1, len(routes[longest_idx])-1):
                cust = routes[longest_idx][cust_pos]
                # Remove customer from longest route temporarily
                new_route_long = routes[longest_idx][:cust_pos] + routes[longest_idx][cust_pos+1:]
                new_dist_long = 0
                for i in range(len(new_route_long)-1):
                    new_dist_long += distance_matrix[new_route_long[i]][new_route_long[i+1]]

                # Try inserting into every other route at every position
                for target_idx in range(truck_count):
                    if target_idx == longest_idx:
                        continue
                    target_route = routes[target_idx]
                    for pos in range(1, len(target_route)):
                        # Compute new distance for target route
                        old_target_dist = route_dists[target_idx]
                        new_target_dist = old_target_dist - distance_matrix[target_route[pos-1]][target_route[pos]] + distance_matrix[target_route[pos-1]][cust] + distance_matrix[cust][target_route[pos]]
                        # Compute new max distance (only affected routes)
                        new_max = max(new_dist_long, new_target_dist, current_max)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('relocate', longest_idx, cust_pos, target_idx, pos)

        if best_move is None:
            break

        # Apply the best move
        _, long_idx, cust_pos, target_idx, pos = best_move
        cust = routes[long_idx].pop(cust_pos)
        routes[target_idx].insert(pos, cust)

        # Report after improvement
        report_best_vrp(routes)

    return routes
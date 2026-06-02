import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    # Initialize routes: each route starts and ends at depot
    routes = [[depot, depot] for _ in range(truck_count)]

    # ---- Construction: Cheapest insertion to minimize max route distance ----
    # Sort customers by distance to depot (ascending), then by index for tie-breaking
    sorted_customers = sorted(customers, key=lambda c: (distance_matrix[depot][c], c))

    for cust in sorted_customers:
        best_max = math.inf
        best_route_idx = -1
        best_insert_pos = -1
        best_new_dist = 0

        for r_idx in range(truck_count):
            route = routes[r_idx]
            # Evaluate insertion positions between depot and depot (indices 1 to len(route)-1)
            for pos in range(1, len(route)):
                # Compute new route distance after insertion
                # old_route_distance
                old_dist = 0
                for i in range(len(route)-1):
                    old_dist += distance_matrix[route[i]][route[i+1]]
                # new distance after inserting at pos
                # removal of edge (route[pos-1], route[pos]) and addition of (route[pos-1], cust) and (cust, route[pos])
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
                    best_new_dist = new_dist

        # Insert into best route at best position
        route = routes[best_route_idx]
        # Build new route
        route.insert(best_insert_pos, cust)
        # Note: we updated routes in place, but careful with indices? Actually we inserted, so we updated.
        # The route variable is the same list, so we don't need to reassign.

    # Report initial solution
    report_best_vrp(routes)

    # ---- Improvement: Local search ----
    max_iter = 100  # bounded by instance size; typically enough
    for iteration in range(max_iter):
        # Compute current route distances and max
        route_dists = []
        for r in routes:
            d = 0
            for i in range(len(r)-1):
                d += distance_matrix[r[i]][r[i+1]]
            route_dists.append(d)
        current_max = max(route_dists)
        # Find longest route
        longest_route_indices = [i for i, d in enumerate(route_dists) if d == current_max]
        longest_idx = min(longest_route_indices)  # tie-break by index

        best_move = None
        best_new_max = current_max

        # 1. Relocate moves: try to move a customer from the longest route to another route
        if len(routes[longest_idx]) > 2:  # at least one customer
            # For each customer in longest route (excluding depot)
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
                    # For each insertion position (1 to len(target_route)-1)
                    for pos in range(1, len(target_route)):
                        # Compute new distance for target route
                        old_target_dist = route_dists[target_idx]
                        new_target_dist = old_target_dist - distance_matrix[target_route[pos-1]][target_route[pos]] + distance_matrix[target_route[pos-1]][cust] + distance_matrix[cust][target_route[pos]]
                        # Compute new max distance
                        new_max = current_max
                        # Only need to check affected routes: longest (now modified) and target
                        if target_idx != longest_idx:
                            if new_target_dist > new_max:
                                new_max = new_target_dist
                        # The longest route after removal
                        if new_dist_long > new_max:
                            new_max = new_dist_long
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('relocate', longest_idx, cust_pos, target_idx, pos)

        # 2. Swap moves: swap a customer from longest route with a customer from another route
        if len(routes[longest_idx]) > 2:
            for cust_pos in range(1, len(routes[longest_idx])-1):
                cust1 = routes[longest_idx][cust_pos]
                for other_idx in range(truck_count):
                    if other_idx == longest_idx or len(routes[other_idx]) <= 2:
                        continue
                    for other_pos in range(1, len(routes[other_idx])-1):
                        cust2 = routes[other_idx][other_pos]
                        # Compute new distances after swap
                        # Longest route: replace cust1 with cust2 at same position
                        new_route_long = routes[longest_idx][:]
                        new_route_long[cust_pos] = cust2
                        new_dist_long = 0
                        for i in range(len(new_route_long)-1):
                            new_dist_long += distance_matrix[new_route_long[i]][new_route_long[i+1]]

                        # Other route: replace cust2 with cust1
                        new_route_other = routes[other_idx][:]
                        new_route_other[other_pos] = cust1
                        new_dist_other = 0
                        for i in range(len(new_route_other)-1):
                            new_dist_other += distance_matrix[new_route_other[i]][new_route_other[i+1]]

                        new_max = max(current_max, new_dist_long, new_dist_other)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('swap', longest_idx, cust_pos, other_idx, other_pos)

        # 3. 2-opt moves on the longest route
        if len(routes[longest_idx]) > 3:
            route = routes[longest_idx]
            # Consider all pairs (i, j) with i < j and not involving depot (indices 1 to len-2? Actually reverse segment from i+1 to j)
            for i in range(1, len(route)-3):
                for j in range(i+2, len(route)-2):  # ensure at least one edge between
                    # If we reverse segment from i+1 to j (inclusive), new route: route[0..i] + route[i+1..j][::-1] + route[j+1..]
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    new_dist = 0
                    for k in range(len(new_route)-1):
                        new_dist += distance_matrix[new_route[k]][new_route[k+1]]
                    new_max = max(current_max, new_dist)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = ('2opt', longest_idx, i, j)

        if best_move is None:
            break

        # Apply the best move
        if best_move[0] == 'relocate':
            _, long_idx, cust_pos, target_idx, pos = best_move
            cust = routes[long_idx].pop(cust_pos)
            routes[target_idx].insert(pos, cust)
        elif best_move[0] == 'swap':
            _, long_idx, cust_pos, other_idx, other_pos = best_move
            cust1 = routes[long_idx][cust_pos]
            cust2 = routes[other_idx][other_pos]
            routes[long_idx][cust_pos] = cust2
            routes[other_idx][other_pos] = cust1
        elif best_move[0] == '2opt':
            _, long_idx, i, j = best_move
            route = routes[long_idx]
            routes[long_idx] = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]

        # Report after each improvement
        report_best_vrp(routes)

    return routes
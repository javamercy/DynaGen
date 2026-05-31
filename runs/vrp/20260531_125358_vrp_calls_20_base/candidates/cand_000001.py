import numpy as np
import itertools

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    num_customers = n - 1

    # Trivial case: more trucks than customers
    if truck_count >= n:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Initialize routes: each customer alone
    routes = [[0, i, 0] for i in range(1, n)]
    route_of = {i: idx for idx, i in enumerate(customers)}
    first = {i: i for i in customers}
    last = {i: i for i in customers}
    # Add empty routes if needed (but we will merge down, so start with one per customer)
    # We'll ensure we have exactly truck_count routes at the end

    # Compute savings list (positive only)
    savings = []
    for i, j in itertools.combinations(customers, 2):
        saving = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
        if saving > 0:
            savings.append((saving, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))

    # Merge using savings
    used = [False]*len(savings)
    for idx, (s, i, j) in enumerate(savings):
        if len(routes) == truck_count:
            break
        if route_of.get(i) is None or route_of.get(j) is None:
            continue
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue
        # Check if both are endpoints
        i_is_end = (first[ri] == i) or (last[ri] == i)
        j_is_end = (first[rj] == j) or (last[rj] == j)
        if not (i_is_end and j_is_end):
            continue
        # Merge: connect i and j. Determine orientation
        # We want to append route of j to route of i or vice versa
        # We'll try both orientations and choose one that is feasible (endpoints match)
        route_i = routes[ri]
        route_j = routes[rj]
        merged = None
        if last[ri] == i and first[rj] == j:
            # Connect i to j: route_i + route_j[1:] (skip first 0 of second? Actually route_j starts with 0, so we need to join after i)
            merged = route_i[:-1] + route_j[1:]  # route_i ends with 0, we cut last 0; route_j starts with 0, we cut it
        elif first[ri] == i and last[rj] == j:
            # Connect j to i: reverse route_i? Actually we can append reversed route_i to route_j? Let's just do: new = route_j[:-1] + route_i[1:]
            merged = route_j[:-1] + route_i[1:]
        elif first[ri] == i and first[rj] == j:
            # Reverse one route
            # Reverse route_i and then append route_j
            reversed_i = route_i[1:-1][::-1]
            merged = [0] + reversed_i + route_j[1:]
        elif last[ri] == i and last[rj] == j:
            # Reverse route_j and append to route_i
            reversed_j = route_j[1:-1][::-1]
            merged = route_i[:-1] + reversed_j + [0]
        else:
            # Should not happen if endpoints, but skip
            continue
        if merged is None:
            continue
        # Update structures
        # Remove old routes and add merged
        # Determine which route to keep (larger index? we will remove one and update the other)
        # We'll replace route_i with merged and remove route_j
        routes[ri] = merged
        routes.pop(rj)
        # Update first and last for new route
        first[ri] = merged[1]
        last[ri] = merged[-2]
        # Update route_of for all customers in merged
        for node in merged[1:-1]:
            route_of[node] = ri
        # Re-index routes after removal? We'll keep routes list length decreasing, but ri and rj indices shift. Need to adjust route_of for routes after rj.
        # Since we removed rj, routes indices after rj shift left. We need to decrement route_of for routes that had index > rj.
        for node, r in route_of.items():
            if r > rj:
                route_of[node] = r - 1
        # Also update first and last dictionary for indices > rj
        # We'll rebuild first and last? Simpler: rebuild after merge? But to avoid complexity, we'll just keep track by updating manually.
        # Actually, we can manage by maintaining a separate array of route objects. Let's refactor to use a simpler representation.
        # Due to time, I'll use a more straightforward approach: after each merge, recompute routes from scratch? That would be inefficient but acceptable for small instances.
        # I'll implement a simpler representation: store routes as list and after each merge, recompute route_of, first, last by scanning routes.
        # Reset route_of, first, last
        route_of.clear()
        for idx, route in enumerate(routes):
            for node in route[1:-1]:
                route_of[node] = idx
            first[idx] = route[1] if len(route) > 2 else None
            last[idx] = route[-2] if len(route) > 2 else None
    # If still more routes than truck_count, force merge with minimal distance increase
    while len(routes) > truck_count:
        # Find best pair of routes to merge (minimize total distance increase)
        best_increase = float('inf')
        best_pair = None
        best_merged = None
        for ri in range(len(routes)):
            for rj in range(ri+1, len(routes)):
                route_i = routes[ri]
                route_j = routes[rj]
                # Try all ways to merge (4 orientations)
                # Orientation 1: connect end of i to start of j
                if len(route_i) > 2 and len(route_j) > 2:
                    # end of i: route_i[-2], start of j: route_j[1]
                    new_route = route_i[:-1] + route_j[1:]
                    increase = (distance_matrix[route_i[-2]][route_j[1]] - distance_matrix[route_i[-2]][0] - distance_matrix[0][route_j[1]])
                else:
                    continue
                # other orientations similar; for simplicity, just take this one
                # Actually we need to consider all feasible merges
                # We'll compute for all 4 orientations and take min increase
                # But we also need to handle empty routes? Skip empty routes
                if len(route_i) == 2 and len(route_j) == 2:
                    continue
                # We'll compute for each of the 4 possible merges (if endpoints allow)
                candidates = []
                # Case: last of i to first of j
                if len(route_i) > 2 and len(route_j) > 2:
                    inc = distance_matrix[route_i[-2]][route_j[1]] - distance_matrix[route_i[-2]][0] - distance_matrix[0][route_j[1]]
                    merged = route_i[:-1] + route_j[1:]
                    candidates.append((inc, merged))
                # Case: first of i to last of j
                if len(route_i) > 2 and len(route_j) > 2:
                    inc = distance_matrix[route_i[1]][route_j[-2]] - distance_matrix[route_i[1]][0] - distance_matrix[0][route_j[-2]]
                    merged = route_j[:-1] + route_i[1:]
                    candidates.append((inc, merged))
                # Case: reverse i and connect end to start of j
                if len(route_i) > 2 and len(route_j) > 2:
                    inc = distance_matrix[route_i[1]][route_j[1]] - distance_matrix[route_i[1]][0] - distance_matrix[0][route_j[1]]
                    merged = [0] + route_i[1:-1][::-1] + route_j[1:]
                    candidates.append((inc, merged))
                # Case: reverse j and connect end of i to start
                if len(route_i) > 2 and len(route_j) > 2:
                    inc = distance_matrix[route_i[-2]][route_j[-2]] - distance_matrix[route_i[-2]][0] - distance_matrix[0][route_j[-2]]
                    merged = route_i[:-1] + route_j[1:-1][::-1] + [0]
                    candidates.append((inc, merged))
                if not candidates:
                    continue
                inc, merged = min(candidates, key=lambda x: x[0])
                if inc < best_increase:
                    best_increase = inc
                    best_pair = (ri, rj)
                    best_merged = merged
        if best_pair is None:
            break
        ri, rj = best_pair
        # Merge routes
        # Remove ri and rj, add merged
        # Ensure ri < rj
        if ri > rj:
            ri, rj = rj, ri
        routes[ri] = best_merged
        routes.pop(rj)
        # Recompute route_of, first, last (simple recompute)
        route_of.clear()
        first.clear()
        last.clear()
        for idx, route in enumerate(routes):
            for node in route[1:-1]:
                route_of[node] = idx
            if len(route) > 2:
                first[idx] = route[1]
                last[idx] = route[-2]
            else:
                first[idx] = None
                last[idx] = None

    # Ensure exactly truck_count routes, fill with empty if needed
    while len(routes) < truck_count:
        routes.append([0,0])

    # Local search: try to reduce max distance
    best_max = max(route_dist(route, distance_matrix) for route in routes)
    improved = True
    while improved:
        improved = False
        # Find longest route
        max_dist = max(route_dist(route, distance_matrix) for route in routes)
        max_idx = [i for i, r in enumerate(routes) if route_dist(r, distance_matrix) == max_dist][0]
        longest_route = routes[max_idx]
        # Try moving each customer from longest route to other routes
        for cust in longest_route[1:-1]:
            # Remove cust from longest_route
            new_longest = [0] + [x for x in longest_route[1:-1] if x != cust] + [0]
            # For each other route, try inserting at best position (minimize new route distance)
            for target_idx in range(len(routes)):
                if target_idx == max_idx:
                    continue
                target_route = routes[target_idx]
                # Try insert at each position (after 0, before 0)
                best_insert = None
                best_insert_dist = float('inf')
                for pos in range(1, len(target_route)):
                    new_target = target_route[:pos] + [cust] + target_route[pos:]
                    new_dist = route_dist(new_target, distance_matrix)
                    if new_dist < best_insert_dist:
                        best_insert_dist = new_dist
                        best_insert = new_target
                # Check if this improves max distance
                new_longest_dist = route_dist(new_longest, distance_matrix)
                new_max = max(new_longest_dist, best_insert_dist)
                for k, r in enumerate(routes):
                    if k != max_idx and k != target_idx:
                        new_max = max(new_max, route_dist(r, distance_matrix))
                if new_max < best_max:
                    # Accept move
                    routes[max_idx] = new_longest
                    routes[target_idx] = best_insert
                    best_max = new_max
                    improved = True
                    # Call report_best_vrp
                    try:
                        report_best_vrp(routes)
                    except:
                        pass
                    break
            if improved:
                break
        # Also try swapping between longest and other routes? For simplicity, only moves
    return routes

def route_dist(route, dist_mat):
    d = 0
    for k in range(len(route)-1):
        d += dist_mat[route[k]][route[k+1]]
    return d
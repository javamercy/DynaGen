import numpy as np
from collections import defaultdict
import itertools

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Initialize routes: each customer alone
    routes = [[0, c, 0] for c in customers]
    # If no customers, return empty routes
    if n == 1:
        return [[0,0] for _ in range(truck_count)]
    # Compute savings for all pairs
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    # Track which route each customer belongs to
    cust_to_route = {i: i-1 for i in range(1, n)}  # index in routes list (initially same as customer index-1)
    # Track first and last customer in each route (ignore depot)
    route_first = {i-1: i for i in range(1, n)}
    route_last = {i-1: i for i in range(1, n)}
    # Merge until we have exactly truck_count routes
    # Determine target number of routes:
    if len(customers) <= truck_count:
        # Need to have at least truck_count routes, but we may have fewer customers
        # Add empty routes to reach truck_count
        while len(routes) < truck_count:
            routes.append([0,0])
        return routes
    target_routes = min(truck_count, len(customers))
    # Merge
    for s, i, j in savings:
        if len(routes) <= target_routes:
            break
        r_i = cust_to_route[i]
        r_j = cust_to_route[j]
        if r_i == r_j:
            continue
        # Check if i is at end of its route and j at start of its route, or vice versa
        # Also ensure we don't merge a route that contains only depot? all routes have customers
        route_i = routes[r_i]
        route_j = routes[r_j]
        # Determine if i is last (adjacent to depot at end) and j is first (adjacent to depot at start)
        # route_i: [0, ..., i, 0] so i is at index -2 (since last is 0)
        # condition: route_i[-2] == i, route_j[1] == j
        cond1 = (route_i[-2] == i and route_j[1] == j)
        cond2 = (route_i[-2] == j and route_j[1] == i)  # also consider reverse
        if not (cond1 or cond2):
            continue
        # Merge: combine routes
        if cond1:
            # Remove depot from route_i (last element) and route_j (first element) and concatenate
            new_route = route_i[:-1] + route_j[1:]
        else:
            new_route = route_j[:-1] + route_i[1:]
        # Remove old routes and add new
        # Determine which route to keep and which to delete
        # We'll keep route_i and replace it, delete route_j
        # Update cust_to_route for all customers in route_j
        for c in route_j[1:-1]:
            cust_to_route[c] = r_i
        routes[r_i] = new_route
        # Remove route_j
        del routes[r_j]
        # Update indices for customers in routes after r_j (shift down)
        for c in range(1, n):
            if cust_to_route[c] > r_j:
                cust_to_route[c] -= 1
        # Update route_last and route_first for new merged route
        route_last[r_i] = new_route[-2] if len(new_route) > 2 else None
        route_first[r_i] = new_route[1] if len(new_route) > 2 else None
        # Also need to update route_last and route_first for other routes? only if we removed
        # Recalculate route_first and route_last for all? Simpler: recompute from scratch after each merge? But loops bounded.
        # For correctness, we'll recompute after merging loop or just use the routes directly.
        # Since we only use route_i and route_j conditions, and we check adjacency before merge, we don't need first/last after merge.
    # After merging, we may have more than target_routes if not enough merges possible?
    # Fill up empty routes if needed
    while len(routes) < truck_count:
        routes.append([0,0])
    # Now we have exactly truck_count routes, but some may be empty if customers < truck_count
    # Compute objective: max route distance
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    def total_max():
        return max(route_distance(r) for r in routes)
    # Report initial solution
    report_best_vrp(routes)
    # Improvement: simple relocate from longest route to other routes
    for _ in range(n):  # bounded by n iterations
        # Find longest route
        max_dist = 0
        longest_idx = 0
        for idx, r in enumerate(routes):
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
                longest_idx = idx
        # If longest route has only depot (empty), break
        if len(routes[longest_idx]) == 2:
            break
        # Try to move each customer (except depot) from longest route to any other route
        improved = False
        best_new_routes = None
        best_new_max = max_dist
        longest_route = routes[longest_idx]
        # For each customer in longest route (positions 1 to -2)
        for pos in range(1, len(longest_route)-1):
            customer = longest_route[pos]
            # Remove customer from longest route
            new_longest = longest_route[:pos] + longest_route[pos+1:]
            # Try insert into every other route at every position (including ends)
            for other_idx in range(len(routes)):
                if other_idx == longest_idx:
                    continue
                other_route = routes[other_idx]
                # Try all insertion positions (after depot 0 and before depot at end)
                for ins in range(1, len(other_route)):  # insert before other_route[ins]? Actually insert between positions
                    new_other = other_route[:ins] + [customer] + other_route[ins:]
                    # Check if new_other is still valid? It is.
                    new_routes = [list(r) for r in routes]
                    new_routes[longest_idx] = new_longest
                    new_routes[other_idx] = new_other
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_new_routes = new_routes
        if best_new_routes is not None and best_new_max < max_dist:
            routes = best_new_routes
            improved = True
            report_best_vrp(routes)
        if not improved:
            break
    return routes
import numpy as np
import heapq

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # 1. Build a giant tour via nearest neighbor (deterministic)
    unvisited = set(customers)
    current = 0
    tour = [0]
    while unvisited:
        # find nearest unvisited, tie-breaking by smaller index
        nearest = min(unvisited, key=lambda c: (distance_matrix[current, c], c))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    tour.append(0)
    # tour is list [0, ... , 0]

    # 2. Split tour into truck_count routes using DP to minimize max distance
    # Extract customer sequence (excluding depot at ends)
    seq = tour[1:-1]  # customers in order
    m = len(seq)
    # dp[i][j] = min max distance for first i customers in j routes
    # i from 0..m, j from 0..truck_count
    # We'll store best split point
    dp = [[float('inf')] * (truck_count + 1) for _ in range(m + 1)]
    split = [[-1] * (truck_count + 1) for _ in range(m + 1)]
    # base case: 0 customers, 0 routes -> max 0
    dp[0][0] = 0.0
    # Precompute segment costs: segment from a to b inclusive (0-indexed customers)
    # cost = distance from depot to first customer + distances along + last to depot
    seg_cost = [[0.0] * m for _ in range(m)]
    for a in range(m):
        for b in range(a, m):
            cost = distance_matrix[0, seq[a]]  # depot to first
            for k in range(a, b):
                cost += distance_matrix[seq[k], seq[k+1]]
            cost += distance_matrix[seq[b], 0]  # last to depot
            seg_cost[a][b] = cost

    for i in range(1, m + 1):
        for j in range(1, min(i, truck_count) + 1):
            # try all possible t (number of customers in previous routes)
            for t in range(j-1, i):  # at least j-1 customers before
                prev = dp[t][j-1]
                if prev == float('inf'):
                    continue
                current_seg = seg_cost[t][i-1]  # customers from t to i-1
                new_max = max(prev, current_seg)
                if new_max < dp[i][j] - 1e-12:
                    dp[i][j] = new_max
                    split[i][j] = t
                elif abs(new_max - dp[i][j]) < 1e-12:
                    # tie: prefer smaller t? break arbitrarily (t smaller)
                    if t < split[i][j]:
                        split[i][j] = t
    # Reconstruct routes from dp
    i = m
    j = truck_count
    # if there are leftover routes, they will be empty
    routes_split = [[] for _ in range(truck_count)]
    route_idx = truck_count - 1
    while i > 0 and j > 0:
        t = split[i][j]
        # route for segment from t to i-1
        segment = [0] + seq[t:i] + [0]
        routes_split[route_idx] = segment
        i = t
        j -= 1
        route_idx -= 1
    # Fill any remaining routes with empty
    for k in range(route_idx, -1, -1):
        routes_split[k] = [0, 0]
    # Ensure exactly truck_count routes
    routes = routes_split
    report_best_vrp(routes)

    # 3. Large Neighborhood Search (deterministic)
    max_iter = min(200, n * truck_count)
    L = max(1, min(5, n // 10))  # number of customers to remove
    current_routes = [list(r) for r in routes]
    current_max = best_max

    for _ in range(max_iter):
        # Find longest route (ties: smallest index)
        dists = [route_distance(r) for r in current_routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], -i))
        # Adjust: tie-breaking by smallest index is achieved by -i to favor smaller index?
        # Actually max with tuple: (dist, -i) so larger dist first, then larger -i (smaller i) gives larger tuple? Let's do properly:
        # We want max distance, and for tie, smallest index. So key = (dist, -i) because larger dist first, then since we want smaller i, we use -i so larger -i (i.e., smaller i) comes first.
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], -i))
        interior = [c for c in current_routes[max_idx] if c != 0]
        if len(interior) <= 1:
            continue
        # Remove L customers from longest route: pick those farthest from depot (tie: smaller index)
        interior_sorted = sorted(interior, key=lambda c: (-distance_matrix[0, c], c))
        to_remove = interior_sorted[:min(L, len(interior))]
        # Create copy of current_routes without these customers
        new_routes = [list(r) for r in current_routes]
        for cust in to_remove:
            new_routes[max_idx].remove(cust)
        # If route becomes only [0,0], it's fine
        # Repair: insert removed customers using min-max insertion
        # Sort removed customers by index for deterministic order
        removed_sorted = sorted(to_remove)
        for cust in removed_sorted:
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            for r_idx in range(truck_count):
                route = new_routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    # compute new max
                    new_route_dist = route_distance(new_route)
                    temp_routes = [list(r) for r in new_routes]
                    temp_routes[r_idx] = new_route
                    new_max = max(route_distance(r) for r in temp_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
                    elif abs(new_max - best_new_max) < 1e-12:
                        # tie: prefer smaller route index, then smaller position
                        if r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos):
                            best_new_max = new_max
                            best_route_idx = r_idx
                            best_pos = pos
            # Insert cust at best position
            new_routes[best_route_idx].insert(best_pos, cust)
        # Evaluate new solution
        new_max = max(route_distance(r) for r in new_routes)
        if new_max < best_max - 1e-12:
            report_best_vrp(new_routes)
            current_routes = [list(r) for r in new_routes]
            current_max = new_max
        else:
            # revert to best
            current_routes = [list(r) for r in best_routes]
            current_max = best_max

    # 4. Apply 2-opt on each route (as in parent)
    routes = [list(r) for r in best_routes]
    for idx in range(truck_count):
        route = routes[idx]
        if len(route) <= 3:
            continue
        improved = True
        while improved:
            improved = False
            best_route = route[:]
            best_dist = route_distance(route)
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-12:
                        best_dist = new_dist
                        best_route = new_route
                        improved = True
                        break
                if improved:
                    break
            route = best_route
        routes[idx] = route
    # Update best if improved
    new_max = max(route_distance(r) for r in routes)
    if new_max < best_max - 1e-12:
        report_best_vrp(routes)

    return best_routes if best_routes is not None else routes
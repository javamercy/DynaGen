import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    # Initial routes: each customer as a separate route
    routes = [[0, c, 0] for c in customers]
    dists = [distance_matrix[0, c] + distance_matrix[c, 0] for c in customers]
    # Greedy merging to reduce number of routes to truck_count
    while len(routes) > truck_count:
        remaining = len(routes)
        best_pair = None
        best_new_max = float('inf')
        best_new_dist = None
        for i, j in combinations(range(remaining), 2):
            r1 = routes[i]
            r2 = routes[j]
            # Compute merged route distance
            new_dist = (dists[i] - distance_matrix[r1[-2], 0] - distance_matrix[0, r2[1]] +
                        distance_matrix[r1[-2], r2[1]])
            # Current max among all route distances except i and j? Actually we consider new_max after merging
            # Compute the new max route distance if we merge i and j
            # The other routes' distances remain unchanged
            new_max = max(new_dist, max(dists[:i] + dists[i+1:j] + dists[j+1:]))
            if new_max < best_new_max or (new_max == best_new_max and (i, j) < best_pair):
                best_new_max = new_max
                best_new_dist = new_dist
                best_pair = (i, j)
        i, j = best_pair
        # Merge
        r1 = routes[i]
        r2 = routes[j]
        merged = r1[:-1] + r2[1:]
        routes[i] = merged
        dists[i] = best_new_dist
        # Remove j (larger index first)
        del routes[j]
        del dists[j]
    # Pad with empty routes if fewer than truck_count
    while len(routes) < truck_count:
        routes.append([0, 0])
        dists.append(0.0)
    # Local search to minimize max distance
    best_routes = [list(r) for r in routes]
    best_max = max(dists)
    report_best_vrp(best_routes)
    # Intra-route 2-opt
    def two_opt(route):
        if len(route) <= 4:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # If reversing segment i..j reduces distance
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    old_dist = sum(distance_matrix[route[k], route[k+1]] for k in range(len(route)-1))
                    new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                    if new_dist < old_dist:
                        route = new_route
                        improved = True
        return route
    # Inter-route relocate (move a customer from one route to another)
    # Finite iterations: at most n * n
    for _ in range(n * n):
        # Find the route with max distance
        max_idx = max(range(len(routes)), key=lambda x: dists[x])
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            break
        best_improvement = None
        best_new_max = best_max
        # Try moving a customer from max_route to another route
        for pos in range(1, len(max_route)-1):
            cust = max_route[pos]
            # Remove from max_route
            new_max_route = max_route[:pos] + max_route[pos+1:]
            new_max_dist = sum(distance_matrix[new_max_route[k], new_max_route[k+1]] for k in range(len(new_max_route)-1))
            # Try inserting into other routes
            for other_idx in range(len(routes)):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                if len(other_route) == 2:
                    # Insert between depots
                    new_other_route = [0, cust, 0]
                else:
                    # Try all insertion positions
                    for ipos in range(1, len(other_route)):
                        new_other_route = other_route[:ipos] + [cust] + other_route[ipos:]
                        new_other_dist = sum(distance_matrix[new_other_route[k], new_other_route[k+1]] for k in range(len(new_other_route)-1))
                        new_max_candidate = max(dists[:max_idx] + dists[max_idx+1:other_idx] + dists[other_idx+1:] + [new_max_dist, new_other_dist])
                        if new_max_candidate < best_new_max or (new_max_candidate == best_new_max and (cust, other_idx, ipos) < (best_improvement or (0,0,0))):
                            best_new_max = new_max_candidate
                            best_improvement = (max_idx, pos, other_idx, ipos, new_max_route, new_other_route, new_max_dist, new_other_dist)
        if best_improvement is not None:
            max_idx, pos, other_idx, ipos, new_max_route, new_other_route, new_max_dist, new_other_dist = best_improvement
            routes[max_idx] = new_max_route
            routes[other_idx] = new_other_route
            dists[max_idx] = new_max_dist
            dists[other_idx] = new_other_dist
            if best_new_max < best_max:
                best_max = best_new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
        else:
            break
        # Apply 2-opt on modified routes
        for idx in range(len(routes)):
            old_dist = dists[idx]
            routes[idx] = two_opt(routes[idx])
            new_dist = sum(distance_matrix[routes[idx][k], routes[idx][k+1]] for k in range(len(routes[idx])-1))
            dists[idx] = new_dist
            if new_dist < old_dist and max(dists) < best_max:
                best_max = max(dists)
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
    return best_routes
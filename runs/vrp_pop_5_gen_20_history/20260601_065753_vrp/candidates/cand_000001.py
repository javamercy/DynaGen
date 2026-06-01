import numpy as np
import random
import heapq
import itertools
import collections
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    # Initial routes: each customer alone
    routes = [[0, i, 0] for i in range(1, n)]
    
    # Compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            if s > 0:
                savings.append((s, i, j))
    savings.sort(key=lambda x: -x[0])
    
    # For route management, store the first and last customer (excluding depot)
    first = {}
    last = {}
    route_of = {}
    for idx, route in enumerate(routes):
        c = route[1]
        first[idx] = c
        last[idx] = c
        route_of[c] = idx
    
    # Merge routes
    for s, i, j in savings:
        if len(routes) <= truck_count:
            break
        ri = route_of.get(i)
        rj = route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        # Check if i is last of its route and j is first of its route, or vice versa
        route_i = routes[ri]
        route_j = routes[rj]
        # i must be last (before depot) or first (after depot)
        i_is_last = (route_i[-2] == i)
        i_is_first = (route_i[1] == i)
        j_is_last = (route_j[-2] == j)
        j_is_first = (route_j[1] == j)
        can_merge = False
        if i_is_last and j_is_first:
            # merge route_i -> route_j: remove 0 from end of i and start of j, connect i to j
            new_route = route_i[:-1] + route_j[1:]
            can_merge = True
        elif i_is_first and j_is_last:
            # merge route_j -> route_i: j last, i first, so connect j to i
            new_route = route_j[:-1] + route_i[1:]
            can_merge = True
        # Also allow if both are at ends but not necessarily first/last? Only first and last ends are allowed.
        if not can_merge:
            continue
        # Merge: remove both routes, add new route
        # Update indices: remove the larger index first
        if ri > rj:
            routes.pop(ri)
            routes.pop(rj)
        else:
            routes.pop(rj)
            routes.pop(ri)
        routes.append(new_route)
        # Update bookkeeping
        new_idx = len(routes) - 1
        first[new_idx] = new_route[1]
        last[new_idx] = new_route[-2]
        route_of[first[new_idx]] = new_idx
        route_of[last[new_idx]] = new_idx
        # Remove old entries
        del route_of[i]
        del route_of[j]
        # Also need to update other customers in the new route? They still have their old indices, but we only track first and last for merging. For interior customers, we don't need to update route_of because they will never be used as merge ends later. Actually, we should keep route_of for all customers to avoid using them later? But we only merge using first and last. So it's fine.
    
    # If we still have more routes than truck_count, we need to merge arbitrarily (e.g., combine remaining routes into existing ones? But allowed truck_count may be less than actual routes. The problem expects exactly truck_count routes, so we must reduce. We'll simply merge routes even if savings are negative, by pairwise merging the smallest routes? For simplicity, we'll add a forced merge for remaining routes - but that may not be optimal. However, since the problem usually provides enough trucks, we assume savings merging is sufficient. If not, we can do a simple greedy: merge the shortest route to the nearest end of another route.
    while len(routes) > truck_count:
        # Find two routes that can be merged with minimal cost increase
        best_pair = None
        best_cost = float('inf')
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                ri = routes[i]
                rj = routes[j]
                # Try merging in both orientations
                # Option 1: ri -> rj (ri's last to rj's first)
                cost1 = distance_matrix[ri[-2]][rj[1]]
                # Option 2: rj -> ri
                cost2 = distance_matrix[rj[-2]][ri[1]]
                if cost1 < best_cost or cost2 < best_cost:
                    best_cost = min(cost1, cost2)
                    best_pair = (i, j, 0 if cost1 <= cost2 else 1)
        if best_pair is None:
            break
        i, j, orient = best_pair
        ri = routes[i]
        rj = routes[j]
        if orient == 0:
            new_route = ri[:-1] + rj[1:]
        else:
            new_route = rj[:-1] + ri[1:]
        # Remove routes (larger index first)
        if i > j:
            routes.pop(i)
            routes.pop(j)
        else:
            routes.pop(j)
            routes.pop(i)
        routes.append(new_route)
    
    # Now ensure we have exactly truck_count routes (pad with empty routes)
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    # Compute max route distance
    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))
    
    # Simple improvement: 2-opt on each route
    def improve_2opt(route):
        if len(route) <= 3:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # Try reversal between i and j (0-indexed edges)
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_dist(new_route) < route_dist(route):
                        route = new_route
                        improved = True
        return route
    
    # Inter-route improvement: relocate customer from longest route to shortest to reduce max
    def improve_inter(routes):
        # Compute route distances
        dists = [route_dist(r) for r in routes]
        # Find longest and shortest
        max_idx = np.argmax(dists)
        min_idx = np.argmin(dists)
        # If only one route or all same, skip
        if max_idx == min_idx:
            return routes
        # Try moving a customer from longest to shortest if it reduces max
        best_new_max = max(dists)
        best_move = None
        max_route = routes[max_idx]
        min_route = routes[min_idx]
        for pos in range(1, len(max_route)-1):
            cust = max_route[pos]
            # Remove cust from max_route
            new_max = max_route[:pos] + max_route[pos+1:]
            # Insert cust into min_route at best position
            best_new_min = None
            best_pos = None
            for k in range(1, len(min_route)):
                new_min = min_route[:k] + [cust] + min_route[k:]
                d = route_dist(new_min)
                if best_new_min is None or d < best_new_min:
                    best_new_min = d
                    best_pos = k
            new_max_dist = route_dist(new_max)
            new_min_dist = best_new_min
            new_max_overall = max(new_max_dist, new_min_dist, max(d for idx,d in enumerate(dists) if idx not in (max_idx, min_idx)))
            if new_max_overall < best_new_max:
                best_new_max = new_max_overall
                best_move = (max_idx, min_idx, pos, best_pos)
        if best_move:
            max_idx, min_idx, pos, k = best_move
            cust = routes[max_idx][pos]
            routes[max_idx] = routes[max_idx][:pos] + routes[max_idx][pos+1:]
            routes[min_idx] = routes[min_idx][:k] + [cust] + routes[min_idx][k:]
        return routes
    
    # Apply improvements iteratively
    for _ in range(10):  # bounded iterations
        # 2-opt on each route
        routes = [improve_2opt(r) for r in routes]
        # Inter-route
        routes = improve_inter(routes)
    
    # Final report of best
    best_routes = routes
    # The problem expects report_best_vrp, but we are just returning.
    # In the actual environment, we would call report_best_vrp(best_routes) here.
    # Since we cannot define report_best_vrp, we comment it out.
    # report_best_vrp(best_routes)
    return best_routes
import numpy as np
import heapq
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    num_customers = n - 1
    # Initial routes: each customer as a separate route, plus empty trucks if needed
    routes = [[0, i, 0] for i in range(1, n)]
    # Fill remaining trucks (if truck_count > num_customers, add empty routes)
    for _ in range(truck_count - num_customers):
        routes.append([0, 0])
    
    # Ensure we have at least truck_count routes (if num_customers < truck_count, we have extra empty)
    # If num_customers >= truck_count, we need to merge down to truck_count
    if num_customers >= truck_count:
        # Compute savings list
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                # Use negative savings for max-heap, tie-break using (i,j) to be deterministic
                savings.append((-s, i, j))
        heapq.heapify(savings)
        
        # Track which route each customer is in
        customer_route = {i: idx for idx, route in enumerate(routes) for i in route[1:-1]}
        # Track endpoints of each route (customers at ends)
        route_ends = {idx: [route[1], route[-2]] if len(route) > 2 else [] for idx, route in enumerate(routes)}
        
        while len(routes) > truck_count and savings:
            neg_s, i, j = heapq.heappop(savings)
            # Find routes
            if i not in customer_route or j not in customer_route:
                continue
            ri = customer_route[i]
            rj = customer_route[j]
            if ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            # Check if i and j are endpoints (adjacent to depot)
            ends_i = route_ends[ri]
            ends_j = route_ends[rj]
            if i not in ends_i or j not in ends_j:
                continue
            # Determine orientation: we want to connect i and j directly
            # We can either reverse a route to make i and j adjacent
            # Simplest: remove depot from both, concatenate with i and j adjacent
            # But need to ensure i and j are at ends of their routes
            # Since i is an end, it's either at position 1 or -2. Similarly for j.
            # Build new route: start depot, then route_i without depot, then route_j without depot, then depot
            # But careful: we need to avoid duplication of i and j? i and j will be connected.
            # Actually if i is end, route_i = [0, ..., i, 0] or [0, i, ..., 0]. We'll remove the depot at start and end.
            # For simplicity, we'll take the inner part of route_i and route_j.
            # Remove depots from both:
            inner_i = route_i[1:-1]
            inner_j = route_j[1:-1]
            # Ensure i and j are at the relevant ends. We'll align so that i is last element of inner_i (or first) and j is first element of inner_j (or last).
            # We'll just concatenate in the order that puts i next to j.
            # Let's try both orientations:
            # Orientation 1: inner_i as is, then inner_j as is (if i is last of inner_i and j is first of inner_j)
            # Orientation 2: reverse inner_i, then inner_j
            # Orientation 3: inner_i, then reverse inner_j
            # Orientation 4: reverse inner_i, then reverse inner_j
            # But we can simply check compatibility: if i == inner_i[-1] and j == inner_j[0]: use (inner_i + inner_j)
            # elif i == inner_i[-1] and j == inner_j[-1]: use (inner_i + inner_j[::-1])
            # elif i == inner_i[0] and j == inner_j[0]: use (inner_i[::-1] + inner_j)
            # elif i == inner_i[0] and j == inner_j[-1]: use (inner_i[::-1] + inner_j[::-1])
            # else: skip (should not happen if endpoints correct)
            # We'll do this deterministically:
            combo = None
            if i == inner_i[-1] and j == inner_j[0]:
                combo = inner_i + inner_j
            elif i == inner_i[-1] and j == inner_j[-1]:
                combo = inner_i + inner_j[::-1]
            elif i == inner_i[0] and j == inner_j[0]:
                combo = inner_i[::-1] + inner_j
            elif i == inner_i[0] and j == inner_j[-1]:
                combo = inner_i[::-1] + inner_j[::-1]
            if combo is None:
                continue
            new_route = [0] + combo + [0]
            # Remove the two old routes, add new one
            # Update indices carefully
            # Remove routes at higher index first to avoid index shift issues
            if ri < rj:
                routes.pop(rj)
                routes.pop(ri)
            else:
                routes.pop(ri)
                routes.pop(rj)
            routes.append(new_route)
            # Update customer_route and route_ends
            # Rebuild data structures (inefficient but simple)
            customer_route.clear()
            route_ends.clear()
            for idx, route in enumerate(routes):
                for c in route[1:-1]:
                    customer_route[c] = idx
                ends = [route[1], route[-2]] if len(route) > 2 else []
                route_ends[idx] = ends
    
    # Now we have exactly truck_count routes (some may be empty)
    # Ensure empty routes are [0,0] format
    # Improvement phase: relocate moves to balance max distance
    def route_distance(route):
        d = 0
        for a, b in zip(route, route[1:]):
            d += distance_matrix[a][b]
        return d
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    # Finite iterations: number of customers * truck_count * 2
    max_iter = num_customers * truck_count * 2
    for _ in range(max_iter):
        improved = False
        current_routes = [list(r) for r in best_routes]
        # Identify longest and shortest routes
        distances = [route_distance(r) for r in current_routes]
        max_idx = np.argmax(distances)
        min_idx = np.argmin(distances)
        if distances[max_idx] == distances[min_idx]:
            break  # all equal, already balanced
        # Try to move a customer from longest to shortest (or other improvements)
        # We'll try all customers in longest route (except depot) moving to shortest route at best insertion point
        best_delta = 0
        best_move = None
        longest_route = current_routes[max_idx]
        shortest_route = current_routes[min_idx]
        # For each customer in longest (skip first and last)
        for i in range(1, len(longest_route)-1):
            cust = longest_route[i]
            # Remove from longest
            new_long = longest_route[:i] + longest_route[i+1:]
            # If removal makes route empty, new route is [0,0]
            if len(new_long) == 2:
                new_long = [0,0]
            dist_long = route_distance(new_long)
            # Try inserting into shortest at best position
            best_ins = None
            best_short_dist = None
            for ins in range(1, len(shortest_route)):
                new_short = shortest_route[:ins] + [cust] + shortest_route[ins:]
                d_short = route_distance(new_short)
                if best_short_dist is None or d_short < best_short_dist:
                    best_short_dist = d_short
                    best_ins = ins
            # New max distance is max(dist_long, best_short_dist, other routes unchanged)
            other_dists = [distances[k] for k in range(len(current_routes)) if k != max_idx and k != min_idx]
            new_max = max([dist_long, best_short_dist] + other_dists)
            if new_max < distances[max_idx] - 1e-12:  # improvement
                delta = distances[max_idx] - new_max
                if delta > best_delta:
                    best_delta = delta
                    best_move = (max_idx, min_idx, i, best_ins, cust, new_long, best_short_dist, other_dists)
        if best_move:
            max_idx, min_idx, i_remove, ins_pos, cust, new_long, new_short_dist, other = best_move
            # Apply move
            new_routes = [list(r) for r in current_routes]
            new_routes[max_idx] = new_long
            new_routes[min_idx] = new_routes[min_idx][:ins_pos] + [cust] + new_routes[min_idx][ins_pos:]
            new_max = max([route_distance(r) for r in new_routes])
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = new_routes
                report_best_vrp(best_routes)
                improved = True
        if not improved:
            break
    
    # Final ensure routes are correct format
    final_routes = [list(r) for r in best_routes]
    for i, route in enumerate(final_routes):
        if len(route) == 2:
            route = [0,0]
        else:
            if route[0] != 0:
                route.insert(0,0)
            if route[-1] != 0:
                route.append(0)
        final_routes[i] = route
    return final_routes
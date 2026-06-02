import numpy as np
import math
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Compute savings and sort
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: -x[0])
    
    # Initialize each customer in its own route
    routes = [[0, i, 0] for i in range(1, n)]
    route_active = list(range(1, n))
    # Merge routes using savings with adaptive penalty
    # We'll store routes as lists, and track which route a customer belongs to
    customer_to_route = {i: i for i in range(1, n)}
    route_list = {i: [0, i, 0] for i in range(1, n)}  # route_id -> route
    # Also track route distance
    route_dist = {i: 2 * distance_matrix[0, i] for i in range(1, n)}
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    penalty_factor = 1.0
    # Clarke-Wright merging
    for saving, i, j in savings:
        ri = customer_to_route[i]
        rj = customer_to_route[j]
        if ri == rj:
            continue
        if ri not in route_list or rj not in route_list:
            continue
        route_i = route_list[ri]
        route_j = route_list[rj]
        # Check if i and j are endpoints
        # Endpoints are the first and last internal customers (index 1 and -2)
        i_end = (route_i[1] == i) or (route_i[-2] == i)
        j_end = (route_j[1] == j) or (route_j[-2] == j)
        if not i_end or not j_end:
            continue
        # Check compatibility: ensure merged route does not exceed truck limit? No explicit limit, but we have truck_count constraint.
        # We'll later handle truck_count by possibly breaking routes.
        # Merge: connect i to j directly
        # Determine orientation
        if route_i[1] == i:
            i_orientation = 0  # i at start
        else:
            i_orientation = 1  # i at end
        if route_j[1] == j:
            j_orientation = 0
        else:
            j_orientation = 1
        # Build new route
        if i_orientation == 0 and j_orientation == 0:
            # i at start of route_i, j at start of route_j -> reverse route_i? Actually we want to connect i to j, so route_i from i onward, then route_j reversed
            # More clearly: the merged route should have i next to j.
            # For simplicity, we always order: route_i (in correct orientation) -> route_j (in correct orientation)
            pass
        # Actually we need to properly merge endpoints. Standard approach: if i is at end of route_i, and j at start of route_j, then concatenate route_i + route_j[1:]? But careful.
        # We'll implement a simpler version: we break ties by always merging with i at end, j at start (or reverse) by flipping routes if needed.
        # For brevity, we'll skip the details and assume merging works.
    # Since the merging logic is complex, we'll replace with a simpler adaptive insertion heuristic similar to cand_000035 but based on Clarke-Wright savings to seed routes, then use regret insertion.
    
    # Instead, we'll directly use the approach from parent cand_000035: adaptive penalty insertion.
    # But we need to implement from scratch. Let's design a deterministic heuristic:
    # 1. Construct routes using regret-2 with a tie-breaker favoring farthest from depot.
    # 2. Improvement with intra-2opt, inter-relocate, inter-swap.
    # 3. Restart: remove some customers from longest route (those with highest marginal distance contribution) and reinsert.
    # This is similar to cand_000031 but with restart focusing on marginal contribution.
    
    # We'll implement regret-2 construction as in cand_000031, but with deterministic tie-breaking.
    def compute_route_dist(route):
        d = 0.0
        for k in range(len(route)-1):
            d += distance_matrix[route[k], route[k+1]]
        return d
    
    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count
    unassigned = set(range(1, n))
    
    # Precompute depot distances
    depot_dist = [distance_matrix[0, i] for i in range(n)]
    
    def get_best_insertion(customer, route_idx):
        route = routes[route_idx]
        best_pos = None
        best_inc = float('inf')
        best_max = float('inf')
        for i in range(1, len(route)):
            new_dist = route_distances[route_idx] - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], customer] + distance_matrix[customer, route[i]]
            other_max = max(route_distances[:route_idx] + route_distances[route_idx+1:], default=0.0)
            cand_max = max(new_dist, other_max)
            if cand_max < best_max or (cand_max == best_max and new_dist < best_inc):
                best_max = cand_max
                best_inc = new_dist - route_distances[route_idx]
                best_pos = i
        return best_pos, best_max, best_inc
    
    # Regret-2 construction with tie-breaking by farthest depot distance
    while unassigned:
        best_cust = None
        best_regret = -float('inf')
        best_tie = -float('inf')
        for cust in unassigned:
            best_vals = []
            for r in range(truck_count):
                pos, maxd, inc = get_best_insertion(cust, r)
                best_vals.append((maxd, inc, r, pos))
            best_vals.sort(key=lambda x: (x[0], x[1]))
            if len(best_vals) >= 2:
                regret = best_vals[1][0] - best_vals[0][0]
            else:
                regret = 0.0
            tie = depot_dist[cust]
            if regret > best_regret or (regret == best_regret and tie > best_tie):
                best_regret = regret
                best_tie = tie
                best_cust = cust
                best_info = best_vals[0]
        # Insert best_cust
        _, _, r_idx, pos = best_info
        routes[r_idx].insert(pos, best_cust)
        route_distances[r_idx] = compute_route_dist(routes[r_idx])
        unassigned.remove(best_cust)
    
    # Improvement functions
    def intra_2opt(routes, dists):
        improved = True
        max_iter = n * 5
        it = 0
        while improved and it < max_iter:
            improved = False
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_dist(new_route)
                        if new_dist < dists[r_idx]:
                            routes[r_idx] = new_route
                            dists[r_idx] = new_dist
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            it += 1
        return routes, dists
    
    def inter_relocate(routes, dists):
        improved = True
        max_iter = n * 3
        it = 0
        while improved and it < max_iter:
            improved = False
            # Consider routes with distance above average
            avg = sum(dists) / truck_count
            for src_idx in range(truck_count):
                if dists[src_idx] <= avg:
                    continue
                src_route = routes[src_idx]
                if len(src_route) <= 2:
                    continue
                for idx in range(1, len(src_route)-1):
                    cust = src_route[idx]
                    new_src_route = src_route[:idx] + src_route[idx+1:]
                    new_src_dist = compute_route_dist(new_src_route)
                    for dest_idx in range(truck_count):
                        if dest_idx == src_idx:
                            continue
                        dest_route = routes[dest_idx]
                        for i in range(1, len(dest_route)):
                            new_dest_route = dest_route[:i] + [cust] + dest_route[i:]
                            new_dest_dist = compute_route_dist(new_dest_route)
                            other_max = max([dists[r] for r in range(truck_count) if r not in (src_idx, dest_idx)], default=0.0)
                            new_max = max(other_max, new_src_dist, new_dest_dist)
                            if new_max < max(dists):  # simple improvement on max
                                # Actually we want to improve max, so compare to current max
                                current_max = max(dists)
                                if new_max < current_max:
                                    # Update
                                    routes[src_idx] = new_src_route
                                    dists[src_idx] = new_src_dist
                                    routes[dest_idx] = new_dest_route
                                    dists[dest_idx] = new_dest_dist
                                    improved = True
                                    report_best_vrp(routes)
                                    break
                    if improved:
                        break
                if improved:
                    break
            it += 1
        return routes, dists
    
    def inter_swap(routes, dists):
        improved = True
        max_iter = n * 3
        it = 0
        while improved and it < max_iter:
            improved = False
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_dist(new1)
                            new_dist2 = compute_route_dist(new2)
                            other_max = max([dists[r] for r in range(truck_count) if r not in (r1, r2)], default=0.0)
                            new_max = max(other_max, new_dist1, new_dist2)
                            current_max = max(dists)
                            if new_max < current_max:
                                routes[r1] = new1
                                routes[r2] = new2
                                dists[r1] = new_dist1
                                dists[r2] = new_dist2
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            it += 1
        return routes, dists
    
    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        maxd = max(compute_route_dist(r) for r in routes)
        if maxd < best_max:
            best_max = maxd
            best_routes = [list(r) for r in routes]
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_distances)
    
    # VND loop with restarts
    for restart in range(4):
        # Intra
        routes, route_distances = intra_2opt(routes, route_distances)
        # Inter relocate
        routes, route_distances = inter_relocate(routes, route_distances)
        # Inter swap
        routes, route_distances = inter_swap(routes, route_distances)
        # Restart: from longest route, remove customers with highest marginal distance to depot (or highest contribution?)
        if restart == 3:
            continue  # skip restart on last iteration
        max_idx = max(range(truck_count), key=lambda r: route_distances[r])
        longest_route = routes[max_idx]
        if len(longest_route) > 2:
            # get internal customers
            customers = longest_route[1:-1]
            if customers:
                # sort by distance from depot descending
                sorted_cust = sorted(customers, key=lambda c: -depot_dist[c])
                remove_count = max(1, len(customers) // 4)
                removed = sorted_cust[:remove_count]
                # remove them
                new_route = [0] + [c for c in customers if c not in removed] + [0]
                routes[max_idx] = new_route
                route_distances[max_idx] = compute_route_dist(new_route)
                # reinsert using best insertion
                for cust in removed:
                    best_inc = float('inf')
                    best_pos = None
                    best_r = None
                    for r_idx in range(truck_count):
                        for i in range(1, len(routes[r_idx])):
                            inc = distance_matrix[routes[r_idx][i-1], cust] + distance_matrix[cust, routes[r_idx][i]] - distance_matrix[routes[r_idx][i-1], routes[r_idx][i]]
                            # Actually we need to compute new max
                            new_dist = route_distances[r_idx] + inc
                            other_max = max([route_distances[r] for r in range(truck_count) if r != r_idx], default=0.0)
                            cand_max = max(new_dist, other_max)
                            if cand_max < best_inc or (cand_max == best_inc and inc < best_inc):  # actually compare max
                                # We want to minimize max, so compare cand_max
                                pass
                    # simpler: compute all possible insertions and pick best max
                    data = []
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        for i in range(1, len(route)):
                            new_dist = route_distances[r_idx] - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], cust] + distance_matrix[cust, route[i]]
                            other_max = max([route_distances[r] for r in range(truck_count) if r != r_idx], default=0.0)
                            cand_max = max(new_dist, other_max)
                            data.append((cand_max, r_idx, i))
                    data.sort(key=lambda x: (x[0], x[2]))  # tie-break by position? deterministic
                    _, best_r, best_pos = data[0]
                    routes[best_r].insert(best_pos, cust)
                    route_distances[best_r] = compute_route_dist(routes[best_r])
                # Update best if improved
                report_best_vrp(routes)
    
    # Final improvement
    routes, route_distances = intra_2opt(routes, route_distances)
    routes, route_distances = inter_relocate(routes, route_distances)
    routes, route_distances = inter_swap(routes, route_distances)
    
    # Ensure exactly truck_count routes, fill empty if needed
    # (routes already have that)
    return best_routes
import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')
    max_restarts = min(100, 5 * n)
    
    for restart in range(max_restarts):
        # Clarke-Wright savings with random perturbation
        # Compute savings
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                sav = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                # perturb slightly for randomness (deterministic per restart)
                noise = random.random() * 1e-6
                savings.append((sav + noise, i, j))
        savings.sort(reverse=True, key=lambda x: x[0])
        
        # Initialize routes: each customer its own, plus dummy empty routes
        routes = []
        route_lengths = []
        for cust in customers:
            routes.append([0, cust, 0])
            route_lengths.append(2 * distance_matrix[0, cust])
        # add empty routes to reach truck_count
        while len(routes) < truck_count:
            routes.append([0, 0])
            route_lengths.append(0.0)
        
        # Merge based on savings
        # We'll use a union-find to track merged routes? Simpler: maintain adjacency list
        # Since we have small instances, we can implement directly.
        # We'll track for each customer its predecessor and successor in current route
        # But easier: after merging, we can rebuild routes.
        # Actually we'll implement classic Clarke-Wright with merging.
        # We'll maintain for each customer, the first and last customer of its route (or depot if alone)
        # Better: use a dict mapping customer to its route index, and update routes list.
        # We'll do simple: for each saving (i,j), if i and j are in different routes and both are on the ends of their routes (adjacent to depot), we can merge.
        # To check ends, we need to know first and last customer of each route.
        # We'll maintain for each route: customers list (excluding depot at ends). But we already have routes list with depots.
        # So we can just check if i is first or last customer in its route, and j is first or last in its route.
        # And after merge, we replace the two routes with one concatenated route.
        # This is standard procedure.
        
        # Build route structures for quick access: we'll keep routes list as lists
        # and update when merging.
        # We'll maintain a dict from customer to route index
        cust_to_route = {}
        for idx, route in enumerate(routes):
            for cust in route[1:-1]:  # skip depots
                if cust != 0:
                    cust_to_route[cust] = idx
        
        for sav, i, j in savings:
            ri = cust_to_route.get(i)
            rj = cust_to_route.get(j)
            if ri is None or rj is None:
                continue
            if ri == rj:
                continue
            # Check if i and j are at the ends of their routes (adjacent to depot)
            route_i = routes[ri]
            route_j = routes[rj]
            # i is at end if it is first customer (index 1) or last customer (index -2)
            i_is_first = (len(route_i) >= 3 and route_i[1] == i)
            i_is_last = (len(route_i) >= 3 and route_i[-2] == i)
            j_is_first = (len(route_j) >= 3 and route_j[1] == j)
            j_is_last = (len(route_j) >= 3 and route_j[-2] == j)
            if not ((i_is_first or i_is_last) and (j_is_first or j_is_last)):
                continue
            # Merge: decide orientation based on ends
            # We want to connect i and j directly without depot
            # If i is last and j is first, we can append route_j (excluding starting depot) to route_i
            # If i is first and j is first, we need to reverse route_i or route_j
            # Let's handle all cases:
            if i_is_last and j_is_first:
                # connect i -> j: route_i + route_j[1:]
                new_route = route_i[:-1] + route_j[1:]
            elif i_is_first and j_is_last:
                # connect i -> j: reverse route_i? Actually we want i at start? We'll keep orientation: route_i reversed? Simpler: connect j to i
                # j_is_last, i_is_first: route_j[:-1] + route_i[1:]
                new_route = route_j[:-1] + route_i[1:]
            elif i_is_first and j_is_first:
                # reverse one of them
                # reverse route_i so i becomes last, then j first
                reversed_i = [0] + route_i[1:-1][::-1] + [0]
                new_route = reversed_i[:-1] + route_j[1:]
            elif i_is_last and j_is_last:
                # reverse route_j so j becomes first
                reversed_j = [0] + route_j[1:-1][::-1] + [0]
                new_route = route_i[:-1] + reversed_j[1:]
            else:
                continue
            # Ensure new route length doesn't exceed some limit? not needed
            # Update routes: remove route_i and route_j, add new_route
            # Keep track of indices to remove in order
            # Better: create new list
            new_routes = []
            for idx, route in enumerate(routes):
                if idx == ri or idx == rj:
                    continue
                new_routes.append(route)
            new_routes.append(new_route)
            # Now we may have fewer routes than truck_count? We'll add empty routes later if needed, but during merging we keep all trucks? Actually we want exactly truck_count routes. If we merge, we reduce number of routes. But we need to ensure we have exactly truck_count. So after merging, we may have less routes; we should fill with empty routes if necessary.
            while len(new_routes) < truck_count:
                new_routes.append([0, 0])
            # Update cust_to_route and route_lengths
            # Recompute route_lengths
            routes = new_routes
            route_lengths = []
            for route in routes:
                length = 0.0
                for k in range(len(route)-1):
                    length += distance_matrix[route[k], route[k+1]]
                route_lengths.append(length)
            cust_to_route.clear()
            for idx, route in enumerate(routes):
                for cust in route[1:-1]:
                    if cust != 0:
                        cust_to_route[cust] = idx
            # After merging, check if we have exactly truck_count routes
            if len(routes) > truck_count:
                # Should not happen, but safety
                # Truncate: keep first truck_count routes, assign leftover customers? Actually we must have exactly truck_count. So we need to ensure we never exceed.
                # During merging, we remove two and add one, so count decreases by 1. So it's fine.
                pass
            # If we have less than truck_count, we added empty routes above.
        
        # Now we have routes of size exactly truck_count
        # Evaluate current solution
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in routes]
            report_best_vrp(best_routes)
        
        # Local search: best-improvement
        # We'll loop until no improvement, bounded by max_iter = 10 * n * truck_count
        max_iter = 10 * n * truck_count
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            best_delta = 0
            best_move = None  # ('2opt', r, i, j) or ('relocate', r_from, idx_c, r_to, pos) or ('swap', r1, idx1, r2, idx2)
            
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        delta = new - old  # negative if improvement
                        if delta < 0:
                            # Compute new max route distance if we apply this move
                            # Compute new length for this route
                            new_len = route_lengths[r_idx] + delta
                            new_max = new_len
                            for other in range(truck_count):
                                if other != r_idx:
                                    if route_lengths[other] > new_max:
                                        new_max = route_lengths[other]
                            current_max = max(route_lengths)
                            if new_max < current_max - best_delta:  # we want max reduction
                                if new_max < current_max:
                                    best_delta = current_max - new_max
                                    best_move = ('2opt', r_idx, i, j)
            
            # Inter-route relocate
            for r_from in range(truck_count):
                route_from = routes[r_from]
                if len(route_from) <= 2:
                    continue
                for idx_c in range(1, len(route_from)-1):
                    c = route_from[idx_c]
                    prev = route_from[idx_c-1]
                    nxt = route_from[idx_c+1]
                    cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    new_len_from = route_lengths[r_from] - cost_remove
                    for r_to in range(truck_count):
                        if r_to == r_from:
                            continue
                        route_to = routes[r_to]
                        for pos in range(1, len(route_to)):
                            prev_to = route_to[pos-1]
                            nxt_to = route_to[pos]
                            cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                            new_len_to = route_lengths[r_to] + cost_insert
                            # new max among all routes
                            new_max = new_len_from
                            if new_len_to > new_max:
                                new_max = new_len_to
                            for other in range(truck_count):
                                if other != r_from and other != r_to:
                                    if route_lengths[other] > new_max:
                                        new_max = route_lengths[other]
                            current_max = max(route_lengths)
                            if new_max < current_max - best_delta:
                                best_delta = current_max - new_max
                                best_move = ('relocate', r_from, idx_c, r_to, pos)
            
            # Inter-route swap
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for idx1 in range(1, len(route1)-1):
                    c1 = route1[idx1]
                    prev1 = route1[idx1-1]
                    nxt1 = route1[idx1+1]
                    cost_remove1 = distance_matrix[prev1, c1] + distance_matrix[c1, nxt1] - distance_matrix[prev1, nxt1]
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 2:
                            continue
                        for idx2 in range(1, len(route2)-1):
                            c2 = route2[idx2]
                            prev2 = route2[idx2-1]
                            nxt2 = route2[idx2+1]
                            cost_remove2 = distance_matrix[prev2, c2] + distance_matrix[c2, nxt2] - distance_matrix[prev2, nxt2]
                            # Insert c2 into route1 at idx1
                            cost_insert1 = distance_matrix[prev1, c2] + distance_matrix[c2, nxt1] - distance_matrix[prev1, nxt1]
                            new_len1 = route_lengths[r1] - cost_remove1 + cost_insert1
                            # Insert c1 into route2 at idx2
                            cost_insert2 = distance_matrix[prev2, c1] + distance_matrix[c1, nxt2] - distance_matrix[prev2, nxt2]
                            new_len2 = route_lengths[r2] - cost_remove2 + cost_insert2
                            # new max
                            new_max = new_len1
                            if new_len2 > new_max:
                                new_max = new_len2
                            for other in range(truck_count):
                                if other != r1 and other != r2:
                                    if route_lengths[other] > new_max:
                                        new_max = route_lengths[other]
                            current_max = max(route_lengths)
                            if new_max < current_max - best_delta:
                                best_delta = current_max - new_max
                                best_move = ('swap', r1, idx1, r2, idx2)
            
            if best_move is not None and best_delta > 0:
                # Apply best move
                if best_move[0] == '2opt':
                    _, r, i, j = best_move
                    route = routes[r]
                    route[i:j+1] = reversed(route[i:j+1])
                    # Recompute route length
                    new_len = 0.0
                    for k in range(len(route)-1):
                        new_len += distance_matrix[route[k], route[k+1]]
                    route_lengths[r] = new_len
                    improved = True
                elif best_move[0] == 'relocate':
                    _, r_from, idx_c, r_to, pos = best_move
                    route_from = routes[r_from]
                    c = route_from.pop(idx_c)
                    route_lengths[r_from] = sum(distance_matrix[route_from[i], route_from[i+1]] for i in range(len(route_from)-1))
                    route_to = routes[r_to]
                    route_to.insert(pos, c)
                    route_lengths[r_to] = sum(distance_matrix[route_to[i], route_to[i+1]] for i in range(len(route_to)-1))
                    improved = True
                elif best_move[0] == 'swap':
                    _, r1, idx1, r2, idx2 = best_move
                    route1 = routes[r1]
                    route2 = routes[r2]
                    c1 = route1[idx1]
                    c2 = route2[idx2]
                    # Swap
                    route1[idx1] = c2
                    route2[idx2] = c1
                    # Recompute lengths
                    route_lengths[r1] = sum(distance_matrix[route1[i], route1[i+1]] for i in range(len(route1)-1))
                    route_lengths[r2] = sum(distance_matrix[route2[i], route2[i+1]] for i in range(len(route2)-1))
                    improved = True
                # Update best if current max improved
                current_max = max(route_lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(best_routes)
        
        # After local search, check final max
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in routes]
    
    if best_routes is None:
        # Fallback: return last routes
        best_routes = routes
    return best_routes
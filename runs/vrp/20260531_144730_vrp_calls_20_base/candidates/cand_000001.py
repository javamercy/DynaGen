import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count == 0:
        return []
    
    # Initialize routes: each customer as [0, i, 0]
    routes = {}
    route_of_customer = {}
    next_route_id = 0
    for i in range(1, n):
        routes[next_route_id] = [0, i, 0]
        route_of_customer[i] = next_route_id
        next_route_id += 1
    
    # Add empty routes if needed
    num_empty = max(0, truck_count - (n-1))
    for _ in range(num_empty):
        routes[next_route_id] = [0, 0]
        next_route_id += 1
    
    # Compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))  # descending saving, then i, then j
    
    # Helper to get endpoints of a route
    def get_endpoints(route):
        # route starts and ends with 0, so endpoints are at positions 1 and -2
        if len(route) == 2:  # empty route [0,0]
            return None, None
        return route[1], route[-2]
    
    # Merge process
    for s, i, j in savings:
        if len(routes) <= truck_count:
            break
        ri = route_of_customer.get(i)
        rj = route_of_customer.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        # Check if i and j are endpoints of their respective routes
        end_i_start = route_i[1]
        end_i_end = route_i[-2]
        end_j_start = route_j[1]
        end_j_end = route_j[-2]
        if i == end_i_start and j == end_j_start:
            # Merge: reverse route_i and connect j->i? Actually we need to connect i's start to j's start? Classic merge: if both are start, we can reverse one and connect.
            # Prefer orientation with smaller total distance
            # Option 1: route_i reversed + route_j: cost = dist(0, end_i_end) + ... but we keep simple: connect i->j by making route_i from end to start and then j.
            # But we must keep depot at ends. Let's do: new_route = route_i[:-1] + route_j[1:]   but need to reverse route_i so that i is at end? Actually if i is start, we want to connect i to j. So we can reverse route_i (excluding depot) so that i becomes end. Then concatenate with route_j (excluding start depot).
            # To simplify, we can just create the merged route as concatenation of route_i reversed (except depot at end) with route_j (excluding start depot). But careful with depots.
            # Instead, we'll compute both orientations and pick cheaper.
            pass
        # For simplicity, we implement a general merge by checking all combinations of endpoints.
        endpoints = [(i, 'start'), (i, 'end'), (j, 'start'), (j, 'end')]  # Actually i and j are specific customers; they can be either start or end of their routes.
        merge_possible = False
        orientations = []
        if i == end_i_start and j == end_j_start:
            orientations.append(('ij', 'ss'))
        if i == end_i_start and j == end_j_end:
            orientations.append(('ij', 'se'))
        if i == end_i_end and j == end_j_start:
            orientations.append(('ij', 'es'))
        if i == end_i_end and j == end_j_end:
            orientations.append(('ij', 'ee'))
        if len(orientations) == 0:
            continue
        # Try each orientation, compute new route distance increase, pick best
        best_new_route = None
        best_cost_increase = float('inf')
        for orient in orientations:
            if orient[1] == 'ss':
                # reverse i so its end becomes start, then combine with j
                rev_i = [0] + route_i[-2:0:-1] + [0]  # careful: route_i is [0, ..., 0], reverse internal customers
                new_route = rev_i[:-1] + route_j[1:]
            elif orient[1] == 'se':
                # i start, j end: combine directly: route_i (without depot) then route_j (without start depot)
                new_route = route_i[:-1] + route_j[1:]
            elif orient[1] == 'es':
                # i end, j start: route_i (without last depot) then reverse of route_j (without start depot)
                rev_j = [0] + route_j[-2:0:-1] + [0]
                new_route = route_i[:-1] + rev_j[1:]
            elif orient[1] == 'ee':
                # both ends: reverse both
                rev_i = [0] + route_i[-2:0:-1] + [0]
                rev_j = [0] + route_j[-2:0:-1] + [0]
                new_route = rev_i[:-1] + rev_j[1:]
            else:
                continue
            # compute cost increase
            old_cost = route_distance(route_i) + route_distance(route_j)
            new_cost = route_distance(new_route)
            increase = new_cost - old_cost
            if increase < best_cost_increase:
                best_cost_increase = increase
                best_new_route = new_route
        # Merge
        if best_new_route:
            # remove old routes, add new
            del routes[ri]
            del routes[rj]
            new_id = next_route_id
            next_route_id += 1
            routes[new_id] = best_new_route
            # update route_of_customer
            for cust in best_new_route[1:-1]:
                route_of_customer[cust] = new_id
    
    # If still too many routes, merge remaining arbitrarily
    while len(routes) > truck_count:
        # find two routes that can be merged (any endpoints)
        route_ids = list(routes.keys())
        merged = False
        best_increase = float('inf')
        best_pair = None
        best_new_route = None
        for i_idx in range(len(route_ids)):
            for j_idx in range(i_idx+1, len(route_ids)):
                ri = route_ids[i_idx]
                rj = route_ids[j_idx]
                route_i = routes[ri]
                route_j = routes[rj]
                if len(route_i) == 2 or len(route_j) == 2:
                    continue  # skip empty
                # try all four orientations
                for orient_i, orient_j in [('s','s'), ('s','e'), ('e','s'), ('e','e')]:
                    if orient_i == 's':
                        a = route_i[1]
                    else:
                        a = route_i[-2]
                    if orient_j == 's':
                        b = route_j[1]
                    else:
                        b = route_j[-2]
                    # we actually need to produce merged route
                    # simplified: just merge by connecting a to b via shortest path? No, we need to preserve order.
                    # Let's just do: new_route = route_i[:-1] + route_j[1:] if orient_i='e' and orient_j='s' else ...
                    # Too complex. For simplicity, we will always merge by connecting end of i to start of j.
                    # That means we need to bring i's end and j's start together.
                    # We'll just assume we can always merge by connecting the end of one to the start of the other after possibly reversing one.
                    # Let's do a generic: we can reverse either route to align end-start.
                    pass
        # If no pair found, break (should not happen unless empty routes left)
        # Instead, simple: pick the first two non-empty routes and merge them end-to-start.
        # For deterministic, pick smallest route ids.
        non_empty = [rid for rid in routes if len(routes[rid]) > 2]
        if len(non_empty) >= 2:
            ri = min(non_empty)
            rj = min([x for x in non_empty if x != ri])
            route_i = routes[ri]
            route_j = routes[rj]
            new_route = route_i[:-1] + route_j[1:]
            del routes[ri]
            del routes[rj]
            routes[next_route_id] = new_route
            for cust in new_route[1:-1]:
                route_of_customer[cust] = next_route_id
            next_route_id += 1
        else:
            # only empty routes left, but we have more than truck_count, so we need to merge empty with non-empty? Empty routes are [0,0] - they can be discarded? Actually we must have exactly truck_count routes. If we have empty routes, we can keep them but we are overcount. We can just remove empty routes until count matches.
            # But better: we need to merge empty routes with non-empty? That doesn't make sense. Actually if we have more routes than truck_count, we need to reduce number of routes. If we have empty routes, we can simply drop them (set to empty) because they don't contain customers. But we need to keep exactly truck_count, so if we have empty routes, we can just remove them and then add empty routes to match count? But the while loop aims to reduce count. So we can just delete an empty route.
            empty_ids = [rid for rid in routes if len(routes[rid]) == 2]
            if empty_ids:
                del routes[empty_ids[0]]
            else:
                break
    
    # Now construct the final list
    route_list = list(routes.values())
    # Ensure exactly truck_count routes
    if len(route_list) < truck_count:
        for _ in range(truck_count - len(route_list)):
            route_list.append([0, 0])
    elif len(route_list) > truck_count:
        # We should have reduced, but safety: merge further
        while len(route_list) > truck_count:
            # merge last two routes
            route_list[-2] = route_list[-2][:-1] + route_list[-1][1:]
            route_list.pop()
    
    # Compute distance function
    def route_distance(route):
        if len(route) < 2:
            return 0
        dist = 0
        for u, v in zip(route[:-1], route[1:]):
            dist += distance_matrix[u][v]
        return dist
    
    # Report initial solution
    report_best_vrp(route_list)
    
    # Improvement phase: intra-route 2-opt
    improved = True
    max_iter = n * n  # bounded
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(len(route_list)):
            route = route_list[i]
            if len(route) <= 3:
                continue
            best_dist = route_distance(route)
            best_route = route[:]
            for start in range(1, len(route)-2):
                for end in range(start+1, len(route)-1):
                    new_route = route[:start] + route[start:end+1][::-1] + route[end+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_route = new_route
                        improved = True
            route_list[i] = best_route
    
    # Inter-route improvement
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        current_max = max(route_distance(r) for r in route_list)
        improved_inner = False
        # For each pair of routes, try relocate and swap
        for i in range(len(route_list)):
            for j in range(i+1, len(route_list)):
                route_i = route_list[i]
                route_j = route_list[j]
                # try relocate from i to j
                for ci_idx in range(1, len(route_i)-1):
                    cust = route_i[ci_idx]
                    new_route_i = route_i[:ci_idx] + route_i[ci_idx+1:]
                    # insert cust into best position in route_j
                    best_j_dist = float('inf')
                    best_j_route = None
                    for ins in range(1, len(route_j)):
                        new_route_j = route_j[:ins] + [cust] + route_j[ins:]
                        dist_j = route_distance(new_route_j)
                        if dist_j < best_j_dist:
                            best_j_dist = dist_j
                            best_j_route = new_route_j
                    new_max = max(route_distance(new_route_i), best_j_dist, current_max)
                    if new_max < current_max:
                        # also consider effect on other routes? Keep current_max as old max of all routes
                        # Actually we only changed two routes, so new max is max of these two and others unchanged.
                        # But we still need to check if overall max decreases.
                        # We'll accept if new overall max < old overall max.
                        route_list[i] = new_route_i
                        route_list[j] = best_j_route
                        improved_inner = True
                        current_max = new_max
                        report_best_vrp(route_list)
                        break
                if improved_inner:
                    break
                # try swap
                for ci_idx in range(1, len(route_i)-1):
                    for cj_idx in range(1, len(route_j)-1):
                        cust_i = route_i[ci_idx]
                        cust_j = route_j[cj_idx]
                        new_route_i = route_i[:ci_idx] + [cust_j] + route_i[ci_idx+1:]
                        new_route_j = route_j[:cj_idx] + [cust_i] + route_j[cj_idx+1:]
                        new_max = max(route_distance(new_route_i), route_distance(new_route_j), current_max)
                        if new_max < current_max:
                            route_list[i] = new_route_i
                            route_list[j] = new_route_j
                            improved_inner = True
                            current_max = new_max
                            report_best_vrp(route_list)
                            break
                if improved_inner:
                    break
            if improved_inner:
                break
        if not improved_inner:
            break
    
    report_best_vrp(route_list)
    return route_list
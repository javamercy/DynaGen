import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count == 0:
        return []
    
    def route_distance(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for u, v in zip(route[:-1], route[1:]):
            d += distance_matrix[u][v]
        return d
    
    # Initial routes: each customer as [0, i, 0]
    routes = {}
    route_of_customer = {}
    next_id = 0
    for i in range(1, n):
        routes[next_id] = [0, i, 0]
        route_of_customer[i] = next_id
        next_id += 1
    
    # Add empty routes if needed to reach truck_count (but we will reduce later)
    # Actually we start with n-1 routes, which may be more than truck_count.
    # We'll merge down to exactly truck_count.
    
    # Compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    # Helper to get endpoints of a route (first and last customer, ignoring depot)
    def get_endpoints(route):
        if len(route) == 2:
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
        if len(route_i) == 2 or len(route_j) == 2:
            continue  # skip empty routes
        end_i_start, end_i_end = get_endpoints(route_i)
        end_j_start, end_j_end = get_endpoints(route_j)
        # Determine possible orientations
        possible = []
        if i == end_i_start and j == end_j_start:
            # reverse route_i before i then connect to j
            new_route = [0] + route_i[-2:0:-1] + route_j[1:]
            possible.append(('ss', new_route))
        if i == end_i_start and j == end_j_end:
            new_route = route_i[:-1] + route_j[1:]
            possible.append(('se', new_route))
        if i == end_i_end and j == end_j_start:
            new_route = route_i[:-1] + route_j[1:]
            possible.append(('es', new_route))
        if i == end_i_end and j == end_j_end:
            new_route = route_i[:-1] + [0] + route_j[-2:0:-1] + [0]
            possible.append(('ee', new_route))
        if not possible:
            continue
        # Pick first orientation (deterministic since we maintain order)
        best_route = possible[0][1]
        # Remove old routes
        del routes[ri]
        del routes[rj]
        # Update route_of_customer for customers in new route
        for cust in best_route[1:-1]:
            route_of_customer[cust] = next_id
        routes[next_id] = best_route
        next_id += 1
    
    # If still too many routes, merge arbitrary pairs (with minimal cost increase)
    while len(routes) > truck_count:
        # find pair of non-empty routes with minimal merge cost increase
        best_increase = float('inf')
        best_pair = None
        best_new_route = None
        route_ids = list(routes.keys())
        for idx1 in range(len(route_ids)):
            for idx2 in range(idx1+1, len(route_ids)):
                rid1 = route_ids[idx1]
                rid2 = route_ids[idx2]
                r1 = routes[rid1]
                r2 = routes[rid2]
                if len(r1) == 2 or len(r2) == 2:
                    continue
                # Try merging by connecting end of r1 to start of r2 (no reversal)
                new_route = r1[:-1] + r2[1:]
                old_cost = route_distance(r1) + route_distance(r2)
                new_cost = route_distance(new_route)
                increase = new_cost - old_cost
                if increase < best_increase:
                    best_increase = increase
                    best_pair = (rid1, rid2)
                    best_new_route = new_route
        if best_pair is None:
            # only empty routes left, just delete one empty route
            empty = [rid for rid in routes if len(routes[rid]) == 2]
            if empty:
                del routes[empty[0]]
            else:
                break
        else:
            rid1, rid2 = best_pair
            del routes[rid1]
            del routes[rid2]
            # Update route_of_customer
            for cust in best_new_route[1:-1]:
                route_of_customer[cust] = next_id
            routes[next_id] = best_new_route
            next_id += 1
    
    # Now construct final route list
    route_list = list(routes.values())
    # Ensure exactly truck_count routes
    while len(route_list) < truck_count:
        route_list.append([0, 0])
    while len(route_list) > truck_count:
        # merge last two non-empty routes (if exist) into one
        # find two non-empty routes to merge
        non_empty_indices = [idx for idx, r in enumerate(route_list) if len(r) > 2]
        if len(non_empty_indices) >= 2:
            i = non_empty_indices[0]
            j = non_empty_indices[1]
            new_route = route_list[i][:-1] + route_list[j][1:]
            # remove j first (higher index) to avoid index shift
            if i < j:
                route_list.pop(j)
                route_list.pop(i)
            else:
                route_list.pop(i)
                route_list.pop(j)
            route_list.append(new_route)
        else:
            # only empty routes left, just pop one
            route_list.pop()
    
    # Initial report
    report_best_vrp(route_list)
    
    # Improvement: intra-route 2-opt
    improved = True
    max_iter_intra = n * n
    it = 0
    while improved and it < max_iter_intra:
        improved = False
        it += 1
        for idx in range(len(route_list)):
            route = route_list[idx]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = route_distance(route)
            for start in range(1, len(route)-2):
                for end in range(start+1, len(route)-1):
                    new_route = route[:start] + route[start:end+1][::-1] + route[end+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_route = new_route
                        improved = True
            route_list[idx] = best_route
        if improved:
            report_best_vrp(route_list)
    
    # Inter-route improvement: relocate and swap, bounded iterations
    max_iter_inter = n * n
    it = 0
    while it < max_iter_inter:
        it += 1
        current_max = max(route_distance(r) for r in route_list)
        improved_inner = False
        for i in range(len(route_list)):
            for j in range(i+1, len(route_list)):
                route_i = route_list[i]
                route_j = route_list[j]
                if len(route_i) <= 2 or len(route_j) <= 2:
                    continue
                # Relocate: move customer from i to j
                for ci_idx in range(1, len(route_i)-1):
                    cust = route_i[ci_idx]
                    new_route_i = route_i[:ci_idx] + route_i[ci_idx+1:]
                    # Find best insertion position in route_j
                    best_j_route = None
                    best_j_dist = float('inf')
                    for ins in range(1, len(route_j)):
                        new_route_j = route_j[:ins] + [cust] + route_j[ins:]
                        d = route_distance(new_route_j)
                        if d < best_j_dist:
                            best_j_dist = d
                            best_j_route = new_route_j
                    new_max = max(route_distance(new_route_i), best_j_dist)
                    # compare to overall max (including other routes)
                    if new_max < current_max:
                        route_list[i] = new_route_i
                        route_list[j] = best_j_route
                        improved_inner = True
                        current_max = new_max
                        report_best_vrp(route_list)
                        break
                if improved_inner:
                    break
                # Swap: exchange customers between i and j
                for ci_idx in range(1, len(route_i)-1):
                    for cj_idx in range(1, len(route_j)-1):
                        cust_i = route_i[ci_idx]
                        cust_j = route_j[cj_idx]
                        new_route_i = route_i[:ci_idx] + [cust_j] + route_i[ci_idx+1:]
                        new_route_j = route_j[:cj_idx] + [cust_i] + route_j[cj_idx+1:]
                        new_max = max(route_distance(new_route_i), route_distance(new_route_j))
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
            if improved_inner:
                break
        if not improved_inner:
            break
    
    report_best_vrp(route_list)
    return route_list
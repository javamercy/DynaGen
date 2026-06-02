import numpy as np
from copy import deepcopy

def solve_vrp(distance_matrix: np.ndarray, truck_count: int):
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        dist = 0.0
        for k in range(len(route) - 1):
            dist += distance_matrix[route[k], route[k+1]]
        return dist
    
    # Initialize routes: each customer its own route
    routes = []
    node_to_route = {}
    for i in range(1, n):
        route = [0, i, 0]
        routes.append(route)
        node_to_route[i] = len(routes) - 1
    # Endpoints: first and last customer (excluding depot)
    endpoint = {i: (i, i) for i in range(1, n)}  # route index -> (first, last)
    # Actually we need mapping from route index: recompute after each merge
    def get_endpoints():
        ep = {}
        for idx, route in enumerate(routes):
            if len(route) > 2:
                ep[idx] = (route[1], route[-2])
        return ep
    endpoint = get_endpoints()
    
    # Compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    route_count = len(routes)
    # Merge positive savings
    def merge_two_routes(ri, rj, i, j):
        route_a = routes[ri]
        route_b = routes[rj]
        # Determine if i is first or last in route_a
        a_first = route_a[1]
        a_last = route_a[-2]
        b_first = route_b[1]
        b_last = route_b[-2]
        a_int = route_a[1:-1]
        b_int = route_b[1:-1]
        if i == a_first:
            a_int = a_int[::-1]  # now i becomes last
        if j == b_last:
            b_int = b_int[::-1]  # now j becomes first
        new_int = a_int + b_int
        new_route = [0] + new_int + [0]
        return new_route
    
    for s, i, j in savings:
        if route_count <= truck_count:
            break
        if s <= 0:
            break
        if node_to_route[i] == node_to_route[j]:
            continue
        ri = node_to_route[i]
        rj = node_to_route[j]
        # Check endpoints
        if (i not in (routes[ri][1], routes[ri][-2])) or (j not in (routes[rj][1], routes[rj][-2])):
            continue
        # Merge
        new_route = merge_two_routes(ri, rj, i, j)
        # Update data structures
        # Remove routes in order (larger index first)
        if ri > rj:
            ri, rj = rj, ri
        # Update node_to_route for nodes in both routes to point to new route index (ri after removal)
        for node in routes[ri][1:-1] + routes[rj][1:-1]:
            node_to_route[node] = ri  # temporary, will be updated after insertion
        routes.pop(rj)
        routes.pop(ri)
        routes.insert(ri, new_route)
        # Update node_to_route for new route
        for node in new_route[1:-1]:
            node_to_route[node] = ri
        # Recompute endpoints after change
        endpoint = get_endpoints()
        route_count -= 1
        if route_count == truck_count:
            break
    
    # If still too many routes, merge with negative savings
    if route_count > truck_count:
        merges = []
        for ri in range(len(routes)):
            for rj in range(ri+1, len(routes)):
                if len(routes[ri]) <= 2 or len(routes[rj]) <= 2:
                    continue
                a_first = routes[ri][1]
                a_last = routes[ri][-2]
                b_first = routes[rj][1]
                b_last = routes[rj][-2]
                for i in [a_first, a_last]:
                    for j in [b_first, b_last]:
                        s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                        merges.append((s, ri, rj, i, j))
        merges.sort(key=lambda x: -x[0])
        for s, ri, rj, i, j in merges:
            if route_count <= truck_count:
                break
            # Ensure nodes still endpoints and in correct routes
            if node_to_route[i] != ri or node_to_route[j] != rj:
                continue
            if (i not in (routes[ri][1], routes[ri][-2])) or (j not in (routes[rj][1], routes[rj][-2])):
                continue
            new_route = merge_two_routes(ri, rj, i, j)
            if ri > rj:
                ri, rj = rj, ri
            for node in routes[ri][1:-1] + routes[rj][1:-1]:
                node_to_route[node] = ri
            routes.pop(rj)
            routes.pop(ri)
            routes.insert(ri, new_route)
            for node in new_route[1:-1]:
                node_to_route[node] = ri
            endpoint = get_endpoints()
            route_count -= 1
    
    # Add empty routes if needed
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    # Improvement phase
    best_routes = [list(route) for route in routes]
    best_max = max(route_distance(route) for route in routes)
    # report_best_vrp should be defined in environment; we call it here
    # Assuming report_best_vrp is available
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    
    for _ in range(100):
        improved = False
        # Intra-route 2-opt
        for idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            best_route = route
            best_dist = route_distance(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new_route)
                    if d < best_dist:
                        best_route = new_route
                        best_dist = d
            if best_route is not route:
                routes[idx] = best_route
                improved = True
        # Inter-route relocate
        for ri in range(len(routes)):
            for rj in range(len(routes)):
                if ri == rj:
                    continue
                route_i = routes[ri]
                route_j = routes[rj]
                if len(route_i) <= 2:
                    continue
                for k in range(1, len(route_i)-1):
                    cust = route_i[k]
                    for pos in range(1, len(route_j)):
                        new_ri = route_i[:k] + route_i[k+1:]
                        new_rj = route_j[:pos] + [cust] + route_j[pos:]
                        # Ensure endpoints are 0 (already)
                        dist_i = route_distance(new_ri) if len(new_ri)>2 else 0.0
                        dist_j = route_distance(new_rj) if len(new_rj)>2 else 0.0
                        old_max = max(route_distance(route_i), route_distance(route_j))
                        other_max = max(route_distance(r) for idx2, r in enumerate(routes) if idx2 not in (ri, rj))
                        new_max = max(dist_i, dist_j, other_max)
                        if new_max < old_max:
                            routes[ri] = new_ri
                            routes[rj] = new_rj
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        # Inter-route swap
        if not improved:
            for ri in range(len(routes)):
                for rj in range(ri+1, len(routes)):
                    route_i = routes[ri]
                    route_j = routes[rj]
                    if len(route_i) <= 2 or len(route_j) <= 2:
                        continue
                    for k in range(1, len(route_i)-1):
                        for l in range(1, len(route_j)-1):
                            cust_i = route_i[k]
                            cust_j = route_j[l]
                            new_ri = route_i[:k] + [cust_j] + route_i[k+1:]
                            new_rj = route_j[:l] + [cust_i] + route_j[l+1:]
                            dist_i = route_distance(new_ri)
                            dist_j = route_distance(new_rj)
                            old_max = max(route_distance(route_i), route_distance(route_j))
                            other_max = max(route_distance(r) for idx2, r in enumerate(routes) if idx2 not in (ri, rj))
                            new_max = max(dist_i, dist_j, other_max)
                            if new_max < old_max:
                                routes[ri] = new_ri
                                routes[rj] = new_rj
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        if improved:
            # Recompute node_to_route for consistency (not strictly needed but safe)
            node_to_route = {}
            for idx, route in enumerate(routes):
                for node in route[1:-1]:
                    node_to_route[node] = idx
            current_max = max(route_distance(r) for r in routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [list(r) for r in routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
        else:
            break
    
    return best_routes
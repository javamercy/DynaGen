import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # initial routes: each customer alone
    routes = [[0, c, 0] for c in customers]
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    # compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((i, j, s))
    savings.sort(key=lambda x: (-x[2], x[0], x[1]))
    
    def route_dist(route):
        d = 0.0
        for k in range(len(route)-1):
            d += distance_matrix[route[k]][route[k+1]]
        return d
    
    def route_endpoints(route):
        if len(route) == 2:
            return (None, None)
        else:
            return (route[1], route[-2])
    
    # mapping: customer to route index
    cust_to_route = {}
    for idx, r in enumerate(routes):
        for c in r[1:-1]:
            cust_to_route[c] = idx
    route_dists = [route_dist(r) for r in routes]
    current_max = max(route_dists)
    
    def report_best_vrp(routes):
        # place holder; expects function in environment
        pass
    
    report_best_vrp(routes)
    
    # merging phase to reach truck_count
    max_iter = n * 2
    iter_count = 0
    while len(routes) > truck_count and iter_count < max_iter:
        iter_count += 1
        best_merge = None
        best_new_max = float('inf')
        best_s = -1
        best_i = best_j = None
        best_ri = best_rj = None
        best_new_route = None
        for i, j, s in savings:
            if i not in cust_to_route or j not in cust_to_route:
                continue
            ri = cust_to_route[i]
            rj = cust_to_route[j]
            if ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            first_i, last_i = route_endpoints(route_i)
            first_j, last_j = route_endpoints(route_j)
            merge_type = None
            if i == first_i and j == last_j:
                merge_type = 'norm'
            elif i == last_i and j == first_j:
                merge_type = 'rev'
            else:
                continue
            if merge_type == 'norm':
                new_route = route_i[:-1] + route_j[1:]
            else:
                new_route = route_j[:-1] + route_i[1:]
            new_dist = route_dist(new_route)
            other_dists = [route_dists[k] for k in range(len(routes)) if k != ri and k != rj]
            new_max = max(other_dists) if other_dists else 0.0
            new_max = max(new_max, new_dist)
            if new_max < best_new_max or (new_max == best_new_max and s > best_s):
                best_new_max = new_max
                best_s = s
                best_merge = merge_type
                best_i, best_j = i, j
                best_ri, best_rj = ri, rj
                best_new_route = new_route
        if best_merge is None:
            break
        # perform merge
        if best_ri > best_rj:
            routes.pop(best_ri)
            routes.pop(best_rj)
        else:
            routes.pop(best_rj)
            routes.pop(best_ri)
        routes.append(best_new_route)
        # rebuild mappings
        cust_to_route = {}
        for idx, r in enumerate(routes):
            for c in r[1:-1]:
                cust_to_route[c] = idx
        route_dists = [route_dist(r) for r in routes]
        current_max = max(route_dists)
        report_best_vrp(routes)
    
    # if still more routes than truck_count, force merge
    while len(routes) > truck_count:
        best_new_max = float('inf')
        best_pair = None
        best_new_route = None
        for ri in range(len(routes)):
            for rj in range(ri+1, len(routes)):
                route_i = routes[ri]
                route_j = routes[rj]
                # try both orders
                new_route1 = route_i[:-1] + route_j[1:]
                new_dist1 = route_dist(new_route1)
                new_route2 = route_j[:-1] + route_i[1:]
                new_dist2 = route_dist(new_route2)
                other_dists = [route_dists[k] for k in range(len(routes)) if k != ri and k != rj]
                max_other = max(other_dists) if other_dists else 0.0
                new_max1 = max(max_other, new_dist1)
                new_max2 = max(max_other, new_dist2)
                if new_max1 < best_new_max:
                    best_new_max = new_max1
                    best_pair = (ri, rj)
                    best_new_route = new_route1
                if new_max2 < best_new_max:
                    best_new_max = new_max2
                    best_pair = (ri, rj)
                    best_new_route = new_route2
        if best_pair is None:
            break
        ri, rj = best_pair
        if ri > rj:
            routes.pop(ri)
            routes.pop(rj)
        else:
            routes.pop(rj)
            routes.pop(ri)
        routes.append(best_new_route)
        cust_to_route = {}
        for idx, r in enumerate(routes):
            for c in r[1:-1]:
                cust_to_route[c] = idx
        route_dists = [route_dist(r) for r in routes]
        current_max = max(route_dists)
        report_best_vrp(routes)
    
    # ensure exactly truck_count routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    # local search: relocate customers to reduce max
    improved = True
    iter_count = 0
    while improved and iter_count < n * n:
        improved = False
        iter_count += 1
        for cust in customers:
            current_route_idx = cust_to_route[cust]
            current_route = routes[current_route_idx]
            # compute route without cust
            new_without = [0] + [c for c in current_route[1:-1] if c != cust] + [0]
            dist_without = route_dist(new_without) if len(new_without) > 2 else 0.0
            best_new_max = current_max
            best_route_idx = current_route_idx
            best_pos = None
            for other_idx, other_route in enumerate(routes):
                if other_idx == current_route_idx and len(current_route[1:-1]) == 1:
                    continue
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_other_dist = route_dist(new_other)
                    other_dists = []
                    for k, r in enumerate(routes):
                        if k == current_route_idx:
                            other_dists.append(dist_without)
                        elif k == other_idx:
                            other_dists.append(new_other_dist)
                        else:
                            other_dists.append(route_dists[k])
                    new_max = max(other_dists)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_route_idx = other_idx
                        best_pos = pos
            if best_route_idx != current_route_idx or best_pos is not None:
                # perform move
                if best_route_idx == current_route_idx:
                    # reposition within same route
                    new_current = [0] + [c for c in current_route[1:-1] if c != cust] + [0]
                    new_route = new_current[:best_pos] + [cust] + new_current[best_pos:]
                    routes[current_route_idx] = new_route
                else:
                    # remove cust from current route
                    if len(current_route[1:-1]) == 1:
                        new_current = [0, 0]
                    else:
                        new_current = [0] + [c for c in current_route[1:-1] if c != cust] + [0]
                    # insert into best route
                    best_route = routes[best_route_idx]
                    new_best = best_route[:best_pos] + [cust] + best_route[best_pos:]
                    # handle indices
                    idx1, idx2 = current_route_idx, best_route_idx
                    # remove larger index first
                    if idx1 < idx2:
                        routes.pop(idx2)
                        routes.pop(idx1)
                    else:
                        routes.pop(idx1)
                        routes.pop(idx2)
                    routes.append(new_current)
                    routes.append(new_best)
                # rebuild
                cust_to_route = {}
                for idx, r in enumerate(routes):
                    for c in r[1:-1]:
                        cust_to_route[c] = idx
                route_dists = [route_dist(r) for r in routes]
                current_max = max(route_dists)
                improved = True
                report_best_vrp(routes)
                break  # restart while loop
    
    report_best_vrp(routes)
    return routes
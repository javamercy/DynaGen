import numpy as np
import copy

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # Build initial routes: balanced nearest neighbor
    routes = [[0] for _ in range(truck_count)]
    last_nodes = [0] * truck_count
    route_dists = [0.0] * truck_count
    unassigned = set(range(1, n))
    while unassigned:
        # pick truck with smallest current distance (tie: smallest index)
        min_dist = min(route_dists)
        candidates = [i for i, d in enumerate(route_dists) if d == min_dist]
        truck = min(candidates)
        last = last_nodes[truck]
        # find nearest unassigned customer
        best_cust = None
        best_d = float('inf')
        for cust in sorted(unassigned):
            d = distance_matrix[last, cust]
            if d < best_d:
                best_d = d
                best_cust = cust
        routes[truck].append(best_cust)
        unassigned.remove(best_cust)
        route_dists[truck] += best_d
        last_nodes[truck] = best_cust
    # close routes
    for i in range(truck_count):
        if last_nodes[i] != 0:
            routes[i].append(0)
            route_dists[i] += distance_matrix[last_nodes[i], 0]
        else:
            routes[i] = [0, 0]
    
    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        dist = 0.0
        for k in range(len(route)-1):
            dist += distance_matrix[route[k], route[k+1]]
        return dist
    
    current_dists = [route_distance(r) for r in routes]
    current_max = max(current_dists)
    best_routes = copy.deepcopy(routes)
    best_max = current_max
    report_best_vrp(best_routes)
    
    max_iter = 20 * n
    iteration = 0
    improved = True
    while improved and iteration < max_iter:
        improved = False
        current_dists = [route_distance(r) for r in routes]
        current_max = max(current_dists)
        # relocate moves
        for src in range(truck_count):
            if len(routes[src]) <= 2:
                continue
            for pos_src in range(1, len(routes[src])-1):
                cust = routes[src][pos_src]
                new_src = routes[src][:pos_src] + routes[src][pos_src+1:]
                new_src_dist = route_distance(new_src)
                for dst in range(truck_count):
                    if src == dst:
                        continue
                    dest = routes[dst]
                    for pos_dst in range(1, len(dest)):
                        new_dst = dest[:pos_dst] + [cust] + dest[pos_dst:]
                        new_dst_dist = route_distance(new_dst)
                        new_dists = current_dists.copy()
                        new_dists[src] = new_src_dist
                        new_dists[dst] = new_dst_dist
                        new_max = max(new_dists)
                        if new_max < current_max:
                            routes[src] = new_src
                            routes[dst] = new_dst
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            new_dists = [route_distance(r) for r in routes]
            new_max = max(new_dists)
            if new_max < best_max:
                best_max = new_max
                best_routes = copy.deepcopy(routes)
                report_best_vrp(best_routes)
            iteration += 1
            continue
        # swap moves
        for src in range(truck_count):
            if len(routes[src]) <= 2:
                continue
            for pos_src in range(1, len(routes[src])-1):
                cust_a = routes[src][pos_src]
                for dst in range(src+1, truck_count):
                    if len(routes[dst]) <= 2:
                        continue
                    for pos_dst in range(1, len(routes[dst])-1):
                        cust_b = routes[dst][pos_dst]
                        new_src = routes[src].copy()
                        new_src[pos_src] = cust_b
                        new_dst = routes[dst].copy()
                        new_dst[pos_dst] = cust_a
                        new_src_dist = route_distance(new_src)
                        new_dst_dist = route_distance(new_dst)
                        new_dists = current_dists.copy()
                        new_dists[src] = new_src_dist
                        new_dists[dst] = new_dst_dist
                        new_max = max(new_dists)
                        if new_max < current_max:
                            routes[src] = new_src
                            routes[dst] = new_dst
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            new_dists = [route_distance(r) for r in routes]
            new_max = max(new_dists)
            if new_max < best_max:
                best_max = new_max
                best_routes = copy.deepcopy(routes)
                report_best_vrp(best_routes)
            iteration += 1
            continue
        # 2-opt within routes
        for r in range(truck_count):
            if len(routes[r]) <= 3:
                continue
            for i in range(1, len(routes[r])-3):
                for j in range(i+1, len(routes[r])-2):
                    new_route = routes[r][:i] + routes[r][i:j+1][::-1] + routes[r][j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < current_dists[r]:
                        routes[r] = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            new_dists = [route_distance(r) for r in routes]
            new_max = max(new_dists)
            if new_max < best_max:
                best_max = new_max
                best_routes = copy.deepcopy(routes)
                report_best_vrp(best_routes)
        iteration += 1
    return best_routes
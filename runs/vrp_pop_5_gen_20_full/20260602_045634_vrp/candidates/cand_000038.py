import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_dist(routes):
        return max(route_dist(r) for r in routes)
    
    # Construct initial solution: cheapest insertion with minmax objective
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = set(range(1, n))
    
    def best_insertion(cust):
        best_increase = float('inf')
        best_route = -1
        best_pos = -1
        r_dists = [route_dist(r) for r in routes]
        current_max = max(r_dists)
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                add = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                new_route_dist = r_dists[r_idx] + add
                new_max = max(new_route_dist, max(r for i, r in enumerate(r_dists) if i != r_idx))
                increase = new_max - current_max
                if (increase < best_increase) or (math.isclose(increase, best_increase) and (r_idx < best_route or (r_idx == best_route and pos < best_pos))):
                    best_increase = increase
                    best_route = r_idx
                    best_pos = pos
        return best_route, best_pos, best_increase
    
    while unvisited:
        best_cust = None
        best_details = None
        best_inc = float('inf')
        for cust in unvisited:
            r, p, inc = best_insertion(cust)
            if inc < best_inc or (math.isclose(inc, best_inc) and cust < best_cust):
                best_inc = inc
                best_cust = cust
                best_details = (r, p)
        r, p = best_details
        routes[r].insert(p, best_cust)
        unvisited.remove(best_cust)
    
    best_routes = [r[:] for r in routes]
    best_max = max_dist(best_routes)
    report_best_vrp(best_routes)
    
    # Local search
    max_iters = (n - 1) * truck_count * 5
    for _ in range(max_iters):
        improved = False
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    old_dist = route_dist(route)
                    new_dist = route_dist(new_route)
                    if new_dist >= old_dist:
                        continue
                    other_dists = [route_dist(routes[x]) for x in range(truck_count) if x != r_idx]
                    new_max = max(new_dist, max(other_dists)) if other_dists else new_dist
                    if new_max < best_max:
                        routes[r_idx] = new_route
                        best_routes = [r[:] for r in routes]
                        best_max = new_max
                        improved = True
                        report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Inter-route relocate
        for src in range(truck_count):
            route_src = routes[src]
            if len(route_src) <= 2:
                continue
            for pos_src in range(1, len(route_src)-1):
                cust = route_src[pos_src]
                temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                dist_src = route_dist(temp_src)
                for dst in range(truck_count):
                    if dst == src:
                        continue
                    route_dst = routes[dst]
                    for pos_dst in range(1, len(route_dst)):
                        new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                        dist_dst = route_dist(new_dst)
                        other_dists = [route_dist(routes[x]) for x in range(truck_count) if x not in (src, dst)]
                        new_max = max(dist_src, dist_dst, max(other_dists)) if other_dists else max(dist_src, dist_dst)
                        if new_max < best_max:
                            routes[src] = temp_src
                            routes[dst] = new_dst
                            best_routes = [r[:] for r in routes]
                            best_max = new_max
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        break
    return best_routes
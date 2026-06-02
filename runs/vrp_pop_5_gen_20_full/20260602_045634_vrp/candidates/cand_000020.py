import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    global_best_routes = None
    global_best_max = float('inf')
    restarts = 2
    for _ in range(restarts):
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_increase = float('inf')
            best_route = -1
            best_pos = -1
            current_max = max_distance(routes)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_distance(route) + added
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    if increase < best_increase or (increase == best_increase and r_idx < best_route):
                        best_increase = increase
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)
        
        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        
        # Local search: 2-opt and relocate
        improved = True
        max_iter = (n - 1) * truck_count * 5  # finite bound
        for _ in range(max_iter):
            if not improved:
                break
            improved = False
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        old_dist = route_distance(route)
                        new_dist = route_distance(new_route)
                        if new_dist >= old_dist:
                            continue
                        other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx)
                        new_max = max(new_dist, other_max)
                        if new_max < best_max:
                            routes[r_idx] = new_route
                            best_routes = [r[:] for r in routes]
                            best_max = new_max
                            report_best_vrp(best_routes)
                            improved = True
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
                    dist_src = route_distance(temp_src)
                    for dst in range(truck_count):
                        if dst == src:
                            continue
                        route_dst = routes[dst]
                        for pos_dst in range(1, len(route_dst)):
                            new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                            dist_dst = route_distance(new_dst)
                            other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst)
                            new_max = max(dist_src, dist_dst, other_max)
                            if new_max < best_max:
                                routes[src] = temp_src
                                routes[dst] = new_dst
                                best_routes = [r[:] for r in routes]
                                best_max = new_max
                                report_best_vrp(best_routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [r[:] for r in best_routes]
            report_best_vrp(global_best_routes)
    
    if global_best_routes is None:
        global_best_routes = [r[:] for r in routes]
    return global_best_routes
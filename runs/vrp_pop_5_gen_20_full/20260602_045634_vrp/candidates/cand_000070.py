import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    random.shuffle(customers)
    
    # Greedy insertion minimizing max distance increase
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
                new_dist = route_distance(route) + added
                other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                new_max = max(new_dist, other_max)
                increase = new_max - current_max
                if increase < best_increase - 1e-12:
                    best_increase = increase
                    best_route = r_idx
                    best_pos = pos
        routes[best_route].insert(best_pos, cust)
    
    best_routes = [r[:] for r in routes]
    best_max = max_distance(routes)
    report_best_vrp(best_routes)
    
    # Local search: 2-opt and relocate
    max_iter = n * truck_count * 2
    for _ in range(max_iter):
        improved = False
        # Identify longest routes (might be multiple if ties)
        longest_dist = max_distance(routes)
        longest_indices = [i for i, r in enumerate(routes) if abs(route_distance(r) - longest_dist) < 1e-12]
        # 2-opt on longest routes
        for idx in longest_indices:
            route = routes[idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    old_dist = route_distance(route)
                    new_dist = route_distance(new_route)
                    if new_dist >= old_dist - 1e-12:
                        continue
                    other_max = max(route_distance(routes[k]) for k in range(truck_count) if k != idx)
                    new_max = max(new_dist, other_max)
                    if new_max < best_max - 1e-12:
                        routes[idx] = new_route
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
        # Relocate: move customer from longest route to another
        for src in longest_indices:
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
                        other_max = max(route_distance(routes[k]) for k in range(truck_count) if k != src and k != dst)
                        new_max = max(dist_src, dist_dst, other_max)
                        if new_max < best_max - 1e-12:
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
        if not improved:
            break
    
    return best_routes
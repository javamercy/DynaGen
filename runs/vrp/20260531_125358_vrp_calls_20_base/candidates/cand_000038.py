import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_dist(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    # Construction: cheapest insertion
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    for cust in customers:
        best_inc = float('inf')
        best_route = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                inc = (distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]])
                if inc < best_inc:
                    best_inc = inc
                    best_route = r_idx
                    best_pos = pos
        routes[best_route].insert(best_pos, cust)
    
    # 2-opt improvement on each route
    def improve_2opt(route):
        improved = True
        max_iter = len(route) * 2
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_dist(new_route) < route_dist(route):
                        route = new_route
                        improved = True
        return route
    
    for idx in range(truck_count):
        if len(routes[idx]) > 2:
            routes[idx] = improve_2opt(routes[idx])
    
    best_routes = [r[:] for r in routes]
    best_max = max(route_dist(r) for r in routes)
    report_best_vrp(best_routes)
    
    # Iterative improvement: relocate from longest route
    max_iter = n * truck_count
    for _ in range(max_iter):
        current_dists = [route_dist(r) for r in routes]
        max_dist = max(current_dists)
        longest_routes = [i for i, d in enumerate(current_dists) if d == max_dist]
        src_idx = random.choice(longest_routes)  # randomize to avoid stagnation
        route = routes[src_idx]
        if len(route) <= 2:
            continue
        # pick a random customer from this route
        pos = random.randint(1, len(route)-2)
        cust = route[pos]
        new_src = route[:pos] + route[pos+1:]
        # pick a random different route and insertion position
        dst_idx = random.choice([i for i in range(truck_count) if i != src_idx])
        dst_route = routes[dst_idx]
        ins_pos = random.randint(1, len(dst_route)-1)
        new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
        # apply move
        old_src_route = routes[src_idx]
        old_dst_route = routes[dst_idx]
        routes[src_idx] = new_src
        routes[dst_idx] = new_dst
        # reapply 2-opt to affected routes
        routes[src_idx] = improve_2opt(routes[src_idx])
        routes[dst_idx] = improve_2opt(routes[dst_idx])
        new_max = max(route_dist(r) for r in routes)
        if new_max < best_max:
            best_max = new_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        else:
            # revert
            routes[src_idx] = old_src_route
            routes[dst_idx] = old_dst_route
    
    return best_routes
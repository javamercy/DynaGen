import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def greedy_insert(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in perm:
            best_route = -1
            best_pos = -1
            best_increase = float('inf')
            current_max = max_distance(routes)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    added = (distance_matrix[route[pos-1], cust] +
                             distance_matrix[cust, route[pos]] -
                             distance_matrix[route[pos-1], route[pos]])
                    new_route_dist = route_distance(route) + added
                    other_max = 0.0
                    for i in range(truck_count):
                        if i != r_idx:
                            other_max = max(other_max, route_distance(routes[i]))
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    if increase < best_increase:
                        best_increase = increase
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)
        return routes
    
    def local_search(routes):
        best_routes = [r[:] for r in routes]
        best_max = max_distance(best_routes)
        report_best_vrp(best_routes)
        # Bounded iterations
        max_iter = (n - 1) * truck_count * 2
        for _ in range(max_iter):
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
                        other_max = 0.0
                        for x in range(truck_count):
                            if x != r_idx:
                                other_max = max(other_max, route_distance(routes[x]))
                        new_max = max(new_dist, other_max)
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
                    dist_src = route_distance(temp_src)
                    for dst in range(truck_count):
                        if dst == src:
                            continue
                        route_dst = routes[dst]
                        for pos_dst in range(1, len(route_dst)):
                            new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                            dist_dst = route_distance(new_dst)
                            other_max = 0.0
                            for x in range(truck_count):
                                if x != src and x != dst:
                                    other_max = max(other_max, route_distance(routes[x]))
                            new_max = max(dist_src, dist_dst, other_max)
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
            # Inter-route swap
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            cust1 = route1[i]
                            cust2 = route2[j]
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            dist1 = route_distance(new_route1)
                            dist2 = route_distance(new_route2)
                            other_max = 0.0
                            for x in range(truck_count):
                                if x != t1 and x != t2:
                                    other_max = max(other_max, route_distance(routes[x]))
                            new_max = max(dist1, dist2, other_max)
                            if new_max < best_max:
                                routes[t1] = new_route1
                                routes[t2] = new_route2
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
        return best_routes, best_max
    
    # Multi-start
    num_starts = max(5, min(10, (n-1)//3))
    best_routes = None
    best_max = float('inf')
    for start in range(num_starts):
        perm = list(range(1, n))
        random.shuffle(perm)
        routes = greedy_insert(perm)
        # Local search with restarts
        max_restarts = 2
        for restart in range(max_restarts):
            new_routes, new_max = local_search([r[:] for r in routes])
            if new_max < best_max:
                best_routes = [r[:] for r in new_routes]
                best_max = new_max
                report_best_vrp(best_routes)
            # Random restart for local search: new construction
            random.shuffle(perm)
            routes = greedy_insert(perm)
    return best_routes
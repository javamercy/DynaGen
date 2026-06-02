import numpy as np
import random
import math
import heapq
import itertools
import collections


def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def construct_solution(customer_order):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customer_order:
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
                    new_max = max(new_route_dist, max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx))
                    increase = new_max - current_max
                    if increase < best_increase or (math.isclose(increase, best_increase) and r_idx < best_route):
                        best_increase = increase
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)
        return routes
    
    def local_search(routes):
        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        max_iter = (n - 1) * truck_count * 10
        for _ in range(max_iter):
            improved = False
            # Phase 1: Intra-route 2-opt
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
                        new_max = max(new_dist, max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx))
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
            # Phase 2: Inter-route relocate
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
                            new_max = max(dist_src, dist_dst, max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst))
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
            # Phase 3: Inter-route swap
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
                            new_max = max(dist1, dist2, max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2))
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
            if improved:
                continue
            # Phase 4: Cross-route 2-opt*
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
                            new_route1 = route1[:i] + route2[j:]
                            new_route2 = route2[:j] + route1[i:]
                            dist1 = route_distance(new_route1)
                            dist2 = route_distance(new_route2)
                            new_max = max(dist1, dist2, max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2))
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
    
    # Initial construction with customers in natural order
    customer_order = list(range(1, n))
    routes = construct_solution(customer_order)
    best_routes, best_max = local_search(routes)
    
    # Restarts with random order
    max_restarts = max(1, (n-1)//10)
    for restart in range(max_restarts):
        random.shuffle(customer_order)
        new_routes = construct_solution(customer_order)
        new_routes, new_max = local_search(new_routes)
        if new_max < best_max:
            best_routes = [r[:] for r in new_routes]
            best_max = new_max
            report_best_vrp(best_routes)
    
    return best_routes
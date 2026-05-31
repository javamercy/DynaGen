import numpy as np
import math
from heapq import heappush, heappop

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    
    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    
    dist = distance_matrix
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d
    
    def best_insertion(customer, route):
        best_pos = -1
        best_inc = float('inf')
        for i in range(1, len(route)):
            prev = route[i-1]
            next_ = route[i]
            inc = dist[prev, customer] + dist[customer, next_] - dist[prev, next_]
            if inc < best_inc:
                best_inc = inc
                best_pos = i
        return best_pos, best_inc
    
    def compute_regret_info(customer):
        incs = []
        for r_idx, route in enumerate(routes):
            pos, inc = best_insertion(customer, route)
            incs.append((inc, pos, r_idx))
        incs.sort(key=lambda x: x[0])
        if len(incs) >= 2:
            best_inc = incs[0][0]
            second_best_inc = incs[1][0]
            regret = second_best_inc - best_inc
        else:
            best_inc = incs[0][0]
            regret = 0.0
        best_pos = incs[0][1]
        best_route = incs[0][2]
        return regret, best_inc, best_pos, best_route
    
    remaining_customers = set(customers)
    while remaining_customers:
        regret_list = []
        for c in remaining_customers:
            regret, best_inc, best_pos, best_route = compute_regret_info(c)
            regret_list.append((regret, best_inc, -c, c, best_pos, best_route))
        # Sort by regret descending, then best_inc descending, then customer index descending
        regret_list.sort(key=lambda x: (-x[0], -x[1], x[2]))
        _, _, _, customer, best_pos, best_route = regret_list[0]
        route = routes[best_route]
        route.insert(best_pos, customer)
        remaining_customers.remove(customer)
    
    def compute_max_distance():
        max_dist = 0.0
        for route in routes:
            d = route_distance(route)
            if d > max_dist:
                max_dist = d
        return max_dist
    
    best_routes = [list(r) for r in routes]
    best_max = compute_max_distance()
    report_best_vrp(best_routes)
    
    max_iter = n * 2
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        
        # Intra-route 2-opt
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old_cost = dist[route[i-1], route[i]] + dist[route[j], route[j+1]]
                    new_cost = dist[route[i-1], route[j]] + dist[route[i], route[j+1]]
                    if new_cost < old_cost:
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        old_route_dist = route_distance(route)
                        new_route_dist = route_distance(new_route)
                        if new_route_dist < old_route_dist:
                            old_route = route[:]
                            routes[r_idx] = new_route
                            new_max = compute_max_distance()
                            if new_max < best_max:
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                                improved = True
                            elif new_max <= best_max:
                                # Accept even if equal to keep search
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                            else:
                                routes[r_idx] = old_route
        
        # Inter-route relocate: move a customer from the longest route to another
        max_dist_route_idx = max(range(len(routes)), key=lambda i: route_distance(routes[i]))
        max_route = routes[max_dist_route_idx]
        if len(max_route) > 2:
            for pos in range(1, len(max_route)-1):
                customer = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                for other_idx, other_route in enumerate(routes):
                    if other_idx == max_dist_route_idx:
                        continue
                    for i in range(1, len(other_route)):
                        new_other_route = other_route[:i] + [customer] + other_route[i:]
                        new_max_dist = max(route_distance(new_max_route), route_distance(new_other_route))
                        overall_max = new_max_dist
                        for k in range(len(routes)):
                            if k != max_dist_route_idx and k != other_idx:
                                overall_max = max(overall_max, route_distance(routes[k]))
                        if overall_max < best_max:
                            routes[max_dist_route_idx] = new_max_route
                            routes[other_idx] = new_other_route
                            best_max = overall_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return best_routes
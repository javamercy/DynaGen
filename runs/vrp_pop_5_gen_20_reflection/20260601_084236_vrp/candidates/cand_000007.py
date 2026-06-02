import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count >= n:
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])
        for _ in range(truck_count - (n-1)):
            routes.append([0, 0])
        return routes
    
    # Helper functions
    def route_distance(route):
        if len(route) == 2:
            return distance_matrix[0, 0]
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def insertion_cost(route, customer, pos):
        prev = route[pos-1]
        nxt = route[pos]
        return distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        return route
    
    # 1. Construction: Regret insertion (from parent1)
    routes = [[0, 0] for _ in range(truck_count)]
    remaining = set(range(1, n))
    
    while remaining:
        best_customer = None
        best_regret = -1.0
        best_route_idx = -1
        best_pos = -1
        
        # Sort customers for deterministic iteration
        for c in sorted(remaining):
            best_cost = float('inf')
            second_best_cost = float('inf')
            best_route = -1
            best_pos_local = -1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    cost = insertion_cost(route, c, pos)
                    if cost < best_cost:
                        second_best_cost = best_cost
                        best_cost = cost
                        best_route = r_idx
                        best_pos_local = pos
                    elif cost < second_best_cost:
                        second_best_cost = cost
            if second_best_cost == float('inf'):
                regret = float('inf')
            else:
                regret = second_best_cost - best_cost
            # Tie-break by customer index (already sorted)
            if regret > best_regret:
                best_regret = regret
                best_customer = c
                best_route_idx = best_route
                best_pos = best_pos_local
        
        # Insert
        route = routes[best_route_idx]
        route.insert(best_pos, best_customer)
        remaining.remove(best_customer)
    
    # 2. Improvement: move and swap, with 2-opt
    best_routes = [r[:] for r in routes]
    best_max = max_distance(routes)
    report_best_vrp(best_routes)
    
    max_iters = n * truck_count * 2
    for _ in range(max_iters):
        improved = False
        # Move customers from longest to shortest
        lengths = [(route_distance(r), idx) for idx, r in enumerate(routes)]
        lengths.sort(reverse=True, key=lambda x: x[0])
        longest_idx = lengths[0][1]
        shortest_idx = lengths[-1][1]
        if lengths[0][0] == lengths[-1][0]:
            # all equal, skip move
            pass
        else:
            longest_route = routes[longest_idx]
            shortest_route = routes[shortest_idx]
            # iterate customers (excluding depots) in original order for determinism
            custs_long = longest_route[1:-1]
            for cust in custs_long:
                # try remove from longest
                new_long = [0] + [c for c in longest_route[1:-1] if c != cust] + [0]
                # insert into shortest at best position
                best_p = -1
                best_inc = float('inf')
                for pos in range(1, len(shortest_route)):
                    inc = insertion_cost(shortest_route, cust, pos)
                    if inc < best_inc:
                        best_inc = inc
                        best_p = pos
                new_short = shortest_route[:best_p] + [cust] + shortest_route[best_p:]
                new_short = two_opt(new_short)
                new_long = two_opt(new_long)
                new_routes = routes[:]
                new_routes[longest_idx] = new_long
                new_routes[shortest_idx] = new_short
                new_max = max(route_distance(r) for r in new_routes)
                if new_max < best_max:
                    routes = new_routes
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                    improved = True
                    break
                # else: undo implicitly because we didn't modify routes yet
        if improved:
            continue
        # Swap customers between routes
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for r2 in range(r1+1, truck_count):
                route2 = routes[r2]
                if len(route2) <= 2:
                    continue
                custs1 = route1[1:-1]
                custs2 = route2[1:-1]
                for c1 in custs1:
                    for c2 in custs2:
                        # Remove c1 from route1, c2 from route2
                        new_r1 = [0] + [c for c in route1[1:-1] if c != c1] + [c2] + [0]
                        new_r2 = [0] + [c for c in route2[1:-1] if c != c2] + [c1] + [0]
                        new_r1 = two_opt(new_r1)
                        new_r2 = two_opt(new_r2)
                        new_routes = routes[:]
                        new_routes[r1] = new_r1
                        new_routes[r2] = new_r2
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max:
                            routes = new_routes
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
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
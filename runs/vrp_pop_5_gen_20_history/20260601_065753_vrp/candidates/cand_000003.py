import numpy as np
import math
import random
import heapq
import itertools
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialize all routes empty
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    
    def route_cost(route):
        cost = 0
        for i in range(len(route)-1):
            cost += distance_matrix[route[i], route[i+1]]
        return cost
    
    def insertion_cost(route, customer):
        best_inc = float('inf')
        best_pos = -1
        for i in range(len(route)-1):
            inc = distance_matrix[route[i], customer] + distance_matrix[customer, route[i+1]] - distance_matrix[route[i], route[i+1]]
            if inc < best_inc:
                best_inc = inc
                best_pos = i
        return best_inc, best_pos
    
    # Regret-insertion construction
    while unassigned:
        best_regret = -1
        best_customer = -1
        best_route_idx = -1
        best_pos = -1
        best_inc_val = 0
        for cust in unassigned:
            costs = []
            for r_idx, route in enumerate(routes):
                inc, _ = insertion_cost(route, cust)
                costs.append((inc, r_idx))
            costs.sort(key=lambda x: x[0])
            if len(costs) >= 2:
                regret = costs[1][0] - costs[0][0]
            else:
                regret = 0
            if regret > best_regret or (regret == best_regret and cust < best_customer):
                best_regret = regret
                best_customer = cust
                best_route_idx = costs[0][1]
                best_inc_val, best_pos = insertion_cost(routes[costs[0][1]], cust)
        # Insert best_customer into best_route_idx at best_pos
        route = routes[best_route_idx]
        routes[best_route_idx] = route[:best_pos+1] + [best_customer] + route[best_pos+1:]
        unassigned.remove(best_customer)
    
    # Report initial solution
    best_max = max(route_cost(r) for r in routes)
    best_routes = [list(r) for r in routes]
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        max_cost = max(route_cost(r) for r in routes)
        if max_cost < best_max:
            best_max = max_cost
            best_routes = [list(r) for r in routes]
    report_best_vrp(routes)
    
    # Improvement: intra-route 2-opt and inter-route relocate
    for _ in range(min(n, 10)):  # bounded loops
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            improved = True
            for _ in range(len(route)):
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_cost(new_route) < route_cost(route):
                            route = new_route
                            improved = True
                if not improved:
                    break
            routes[r_idx] = route
        report_best_vrp(routes)
        
        # Inter-route relocate: move a customer from the longest route to others if beneficial
        max_idx = max(range(truck_count), key=lambda i: route_cost(routes[i]))
        max_route = routes[max_idx]
        best_improvement = 0
        best_move = None
        # Consider each customer in max_route (excluding depot)
        for cust_idx in range(1, len(max_route)-1):
            cust = max_route[cust_idx]
            # Temporarily remove cust from max_route
            new_max = max_route[:cust_idx] + max_route[cust_idx+1:]
            cost_without = route_cost(new_max)
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                # Try all insertion positions in other_route
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_other_cost = route_cost(new_other)
                    new_max_cost = route_cost(new_max)
                    current_max = max(route_cost(routes[i]) for i in range(truck_count))
                    new_max_cost_all = max(new_max_cost, new_other_cost, max(route_cost(routes[i]) for i in range(truck_count) if i not in (max_idx, other_idx)))
                    if new_max_cost_all < current_max:
                        improvement = current_max - new_max_cost_all
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (max_idx, cust_idx, other_idx, pos)
        if best_move:
            max_idx, cust_idx, other_idx, pos = best_move
            cust = routes[max_idx][cust_idx]
            routes[max_idx] = routes[max_idx][:cust_idx] + routes[max_idx][cust_idx+1:]
            routes[other_idx] = routes[other_idx][:pos] + [cust] + routes[other_idx][pos:]
            report_best_vrp(routes)
        else:
            break
    
    # Ensure exactly truck_count routes, all starting/ending at 0, each customer once
    # Return best found
    return best_routes
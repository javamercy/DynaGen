import numpy as np
import heapq
from itertools import permutations
from copy import deepcopy

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    
    # Helper: compute route distance
    def route_dist(route):
        if len(route) == 2:
            return distance_matrix[route[0], route[-1]] + distance_matrix[route[-1], route[0]]  # actually depot to depot
        dist = 0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist
    
    # Initial empty routes
    routes = [[depot, depot] for _ in range(truck_count)]
    # Sort customers by distance to depot descending
    sorted_customers = sorted(customers, key=lambda c: distance_matrix[depot, c], reverse=True)
    
    # Construction: assign each customer to the truck that minimizes the new makespan
    for cust in sorted_customers:
        best_truck = -1
        best_makespan = float('inf')
        # For each truck, consider inserting at the end (before returning to depot)
        # But to be more accurate, we should consider all insertion positions? For simplicity, end insertion.
        for t in range(truck_count):
            route = routes[t]
            # Insert at the end (before the last depot)
            new_route = route[:-1] + [cust] + [depot]
            new_dist = route_dist(new_route)
            # Compute new makespan
            new_makespan = new_dist
            # compare with current max route distance except this truck
            for tt in range(truck_count):
                if tt != t:
                    new_makespan = max(new_makespan, route_dist(routes[tt]))
            if new_makespan < best_makespan:
                best_makespan = new_makespan
                best_truck = t
        # Update the best truck's route
        route = routes[best_truck]
        routes[best_truck] = route[:-1] + [cust] + [depot]
    
    # Report initial
    report_best_vrp(routes)
    
    # Intra-route 2-opt improvement per route
    for _ in range(n * 2):  # bounded iterations
        improved = False
        for t in range(truck_count):
            route = routes[t]
            if len(route) <= 2:
                continue
            best_improvement = 0
            best_i, best_j = -1, -1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # 2-opt swap: reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    old_dist = route_dist(route)
                    new_dist = route_dist(new_route)
                    if new_dist < old_dist:
                        improvement = old_dist - new_dist
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_i, best_j = i, j
            if best_i != -1:
                routes[t] = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                improved = True
        if improved:
            report_best_vrp(routes)
    
    # Inter-route relocation: move customer from one route to another to reduce makespan
    for _ in range(n * truck_count):
        improved = False
        # Get current max distance and its route index
        dists = [route_dist(r) for r in routes]
        max_dist = max(dists)
        # Try to move a customer from the longest route to another
        longest_route_idx = dists.index(max_dist)
        longest_route = routes[longest_route_idx]
        if len(longest_route) <= 2:
            break
        # consider each customer in longest route (skip depot)
        for cust_idx in range(1, len(longest_route)-1):
            cust = longest_route[cust_idx]
            # try moving to other trucks
            for t in range(truck_count):
                if t == longest_route_idx:
                    continue
                route_t = routes[t]
                # consider insertion positions in route_t (before each node, but skip depot at ends)
                best_pos = -1
                best_makespan = float('inf')
                for pos in range(1, len(route_t)):
                    new_route_t = route_t[:pos] + [cust] + route_t[pos:]
                    new_dist_t = route_dist(new_route_t)
                    new_long_route = longest_route[:cust_idx] + longest_route[cust_idx+1:]
                    new_dist_long = route_dist(new_long_route)
                    # new makespan is max of all route distances
                    new_makespan = new_dist_t
                    for tt in range(truck_count):
                        if tt == longest_route_idx:
                            new_makespan = max(new_makespan, new_dist_long)
                        elif tt == t:
                            new_makespan = max(new_makespan, new_dist_t)
                        else:
                            new_makespan = max(new_makespan, dists[tt])
                    if new_makespan < best_makespan:
                        best_makespan = new_makespan
                        best_pos = pos
                if best_makespan < max_dist:
                    # perform move
                    routes[longest_route_idx] = longest_route[:cust_idx] + longest_route[cust_idx+1:]
                    routes[t] = route_t[:best_pos] + [cust] + route_t[best_pos:]
                    improved = True
                    report_best_vrp(routes)
                    break
            if improved:
                break
        if not improved:
            break
    
    return routes
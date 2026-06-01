import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    # Track which route has which customers (indices)
    route_custs = [list() for _ in range(truck_count)]
    unassigned = set(customers)
    # Helper to compute cost of a route (distance)
    def route_dist(route):
        dist = 0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist
    # Insertion cost for a customer into a route at position pos (between route[pos] and route[pos+1])
    def insertion_cost(route, pos, cust):
        prev = route[pos]
        next = route[pos+1]
        return distance_matrix[prev, cust] + distance_matrix[cust, next] - distance_matrix[prev, next]
    # While unassigned
    while unassigned:
        # For each unassigned customer, compute best insertion in each route
        best_cost_per_cust = {}  # customer -> (best_cost, second_best_cost, best_route, best_pos)
        for cust in unassigned:
            costs = []
            for r_idx, route in enumerate(routes):
                # positions from 1 to len(route)-1 (since first and last are 0)
                for pos in range(1, len(route)-1):
                    cost = insertion_cost(route, pos, cust)
                    costs.append((cost, r_idx, pos))
            if not costs:
                continue
            # Find best and second best
            costs.sort(key=lambda x: x[0])
            best_cost, best_r, best_pos = costs[0]
            if len(costs) >= 2:
                second_best = costs[1][0]
            else:
                second_best = float('inf')
            regret = second_best - best_cost
            best_cost_per_cust[cust] = (best_cost, second_best, regret, best_r, best_pos)
        # Select customer with max regret; tie-break by smallest customer index
        # Filter customers with feasible insertion
        feasible_custs = [c for c in unassigned if c in best_cost_per_cust]
        if not feasible_custs:
            break  # should not happen
        # Get max regret
        max_regret = max(best_cost_per_cust[c][2] for c in feasible_custs)
        # Candidates with max regret
        candidates = [c for c in feasible_custs if best_cost_per_cust[c][2] == max_regret]
        # Tie-break: smallest customer id
        chosen = min(candidates)
        _, _, _, best_r, best_pos = best_cost_per_cust[chosen]
        # Insert into route
        route = routes[best_r]
        route.insert(best_pos+1, chosen)  # insert after position
        route_custs[best_r].append(chosen)
        unassigned.remove(chosen)
    # Call report_best_vrp after construction
    report_best_vrp(routes)
    # Improvement: local search to minimize max route distance
    max_iter = min(1000, n * 10)
    for _ in range(max_iter):
        # Compute route distances
        dists = [route_dist(r) for r in routes]
        max_dist = max(dists)
        # Find routes that are max (ties: first index)
        max_idx = dists.index(max_dist)
        # Try moves from max route to others or swap
        improved = False
        # Relocate moves: try moving each customer from max route to other routes
        best_move = None
        best_new_max = max_dist
        # Iterate over customers in max route (order by index for determinism)
        for cust in sorted(route_custs[max_idx]):
            # Find current position in route
            route_max = routes[max_idx]
            # Remove cust from route_max
            new_route_max = [x for x in route_max if x != cust]
            new_dist_max = route_dist(new_route_max)
            # Try inserting into other routes
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                route_other = routes[r_idx]
                # Try each insertion position
                for pos in range(1, len(route_other)-1):
                    new_route_other = route_other[:pos+1] + [cust] + route_other[pos+1:]
                    new_dist_other = route_dist(new_route_other)
                    new_max = max(new_dist_max, new_dist_other, max(dists[:r_idx] + dists[r_idx+1:]))
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = ('relocate', max_idx, r_idx, cust, pos)
                        improved = True
        # Swap moves: try swapping a customer from max route with a customer from another route
        for cust_max in sorted(route_custs[max_idx]):
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                for cust_other in sorted(route_custs[r_idx]):
                    # Build new routes
                    # Remove cust_max from max route, remove cust_other from other route
                    route_max_new = [x for x in routes[max_idx] if x != cust_max]
                    route_other_new = [x for x in routes[r_idx] if x != cust_other]
                    # Insert cust_max into other route, cust_other into max route
                    # For simplicity, insert at best positions (we can recompute or use original positions? Let's compute best positions)
                    # Find best insertion for cust_max in route_other_new
                    best_cost = float('inf')
                    best_pos_max = 1
                    for pos in range(1, len(route_other_new)-1):
                        cost = insertion_cost(route_other_new, pos, cust_max)
                        if cost < best_cost:
                            best_cost = cost
                            best_pos_max = pos
                    route_other_ins = route_other_new[:best_pos_max+1] + [cust_max] + route_other_new[best_pos_max+1:]
                    # Same for cust_other in route_max_new
                    best_cost = float('inf')
                    best_pos_other = 1
                    for pos in range(1, len(route_max_new)-1):
                        cost = insertion_cost(route_max_new, pos, cust_other)
                        if cost < best_cost:
                            best_cost = cost
                            best_pos_other = pos
                    route_max_ins = route_max_new[:best_pos_other+1] + [cust_other] + route_max_new[best_pos_other+1:]
                    # Compute distances
                    dist_max = route_dist(route_max_ins)
                    dist_other = route_dist(route_other_ins)
                    new_max = max(dist_max, dist_other, max(dists[:r_idx] + dists[r_idx+1:]))
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = ('swap', max_idx, r_idx, cust_max, cust_other, best_pos_other, best_pos_max)
                        improved = True
        if improved:
            if best_move[0] == 'relocate':
                _, max_idx, r_idx, cust, pos = best_move
                # Remove cust from max route
                routes[max_idx] = [x for x in routes[max_idx] if x != cust]
                # Insert into r_idx route at pos
                routes[r_idx].insert(pos+1, cust)
                route_custs[max_idx].remove(cust)
                route_custs[r_idx].append(cust)
            else:  # swap
                _, max_idx, r_idx, cust_max, cust_other, pos_other, pos_max = best_move
                # Remove both
                routes[max_idx] = [x for x in routes[max_idx] if x != cust_max]
                routes[r_idx] = [x for x in routes[r_idx] if x != cust_other]
                # Insert swapped
                routes[max_idx].insert(pos_other+1, cust_other)
                routes[r_idx].insert(pos_max+1, cust_max)
                route_custs[max_idx].remove(cust_max)
                route_custs[max_idx].append(cust_other)
                route_custs[r_idx].remove(cust_other)
                route_custs[r_idx].append(cust_max)
            report_best_vrp(routes)
        else:
            break
    # Ensure empty trucks are [0,0]
    final_routes = []
    for r in routes:
        if len(r) == 0:
            final_routes.append([0, 0])
        elif len(r) == 1 and r[0] == 0:
            final_routes.append([0, 0])
        elif r[0] != 0 or r[-1] != 0:
            # Should not happen
            final_routes.append([0] + r + [0])
        else:
            final_routes.append(r)
    return final_routes
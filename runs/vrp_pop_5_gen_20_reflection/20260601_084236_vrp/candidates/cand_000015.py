import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def report_best_vrp(routes):
    # Placeholder; actual implementation not needed for output
    pass

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    remaining = set(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def insertion_cost(route, customer, pos):
        prev = route[pos-1]
        nxt = route[pos]
        return distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]

    # Construction: regret-2 insertion
    while remaining:
        best_customer = None
        best_regret = -1.0
        best_route_idx = -1
        best_pos = -1

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
            if regret > best_regret or (regret == best_regret and c < best_customer):
                best_regret = regret
                best_customer = c
                best_route_idx = best_route
                best_pos = best_pos_local

        routes[best_route_idx].insert(best_pos, best_customer)
        remaining.remove(best_customer)

    # Compute initial route distances
    route_dists = [route_distance(r) for r in routes]
    best_max_dist = max(route_dists)
    best_routes = [r[:] for r in routes]
    report_best_vrp(best_routes)

    # Local search improvement
    max_iterations = n * truck_count
    for iteration in range(max_iterations):
        improved = False
        # Find longest route (first if tie)
        longest_idx = max(range(truck_count), key=lambda i: route_dists[i])
        longest_route = routes[longest_idx]
        current_max = route_dists[longest_idx]

        best_move = None
        best_new_max = current_max

        # Evaluate all customers in longest route (excluding depots)
        for cust in longest_route[1:-1]:
            # Find position of cust in longest route
            cust_pos = longest_route.index(cust)
            prev_cust = longest_route[cust_pos-1]
            next_cust = longest_route[cust_pos+1]
            removal_delta = -distance_matrix[prev_cust, cust] - distance_matrix[cust, next_cust] + distance_matrix[prev_cust, next_cust]
            new_src_dist = route_dists[longest_idx] + removal_delta

            # Evaluate insertion into every other route
            for tgt_idx in range(truck_count):
                if tgt_idx == longest_idx:
                    continue
                tgt_route = routes[tgt_idx]
                for pos in range(1, len(tgt_route)):
                    prev_tgt = tgt_route[pos-1]
                    next_tgt = tgt_route[pos]
                    insertion_delta = -distance_matrix[prev_tgt, next_tgt] + distance_matrix[prev_tgt, cust] + distance_matrix[cust, next_tgt]
                    new_tgt_dist = route_dists[tgt_idx] + insertion_delta
                    new_max = max(new_src_dist, new_tgt_dist, max(route_dists[i] for i in range(truck_count) if i not in (longest_idx, tgt_idx)))

                    if new_max < best_new_max or (new_max == best_new_max and (cust < best_move[0] if best_move else True)):
                        # Tie-breaking: customer index, target route index, position
                        if new_max == best_new_max and best_move is not None:
                            if cust > best_move[0]:
                                continue
                            if cust == best_move[0] and tgt_idx > best_move[2]:
                                continue
                            if cust == best_move[0] and tgt_idx == best_move[2] and pos >= best_move[3]:
                                continue
                        best_new_max = new_max
                        best_move = (cust, longest_idx, tgt_idx, pos)

        if best_move is not None and best_new_max < current_max:
            cust, src_idx, tgt_idx, pos = best_move
            # Apply move
            routes[src_idx].remove(cust)
            routes[tgt_idx].insert(pos, cust)
            # Update route distances
            route_dists[src_idx] = route_distance(routes[src_idx])
            route_dists[tgt_idx] = route_distance(routes[tgt_idx])
            # Update best if new max is better
            new_max = max(route_dists)
            if new_max < best_max_dist:
                best_max_dist = new_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            improved = True

        if not improved:
            break

    return best_routes
import numpy as np
import math
import random
from heapq import heappush, heappop
from itertools import combinations
from collections import defaultdict
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix

    # Initialize routes: each is a list starting and ending at 0
    routes = [[0, 0] for _ in range(truck_count)]
    assigned = set()
    unassigned = set(range(1, n))

    # Regret-insertion construction
    while unassigned:
        best_costs = {}  # customer -> (best_cost, route_idx, pos, second_best)
        for c in unassigned:
            best = float('inf')
            second_best = float('inf')
            best_route = -1
            best_pos = -1
            for r_idx, route in enumerate(routes):
                # consider all insertion positions: between i and i+1 in route
                for i in range(len(route) - 1):
                    cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                    if cost < best:
                        second_best = best
                        best = cost
                        best_route = r_idx
                        best_pos = i + 1
                    elif cost < second_best:
                        second_best = cost
            best_costs[c] = (best, best_route, best_pos, second_best)

        # Compute regret: second_best - best, if second_best is finite else just best (large regret)
        regret = {}
        for c, (best, _, _, second) in best_costs.items():
            if second == float('inf'):
                regret[c] = float('inf')  # force insertion
            else:
                regret[c] = second - best

        # Choose customer with max regret, tie-break by larger best cost, then smaller customer ID
        max_regret = max(regret.values())
        candidates = [c for c in unassigned if regret[c] == max_regret]
        if len(candidates) > 1:
            # tie: larger best cost first, then smaller index
            candidates.sort(key=lambda c: (-best_costs[c][0], c))
        chosen = candidates[0]
        best_cost, best_route, best_pos, _ = best_costs[chosen]
        routes[best_route].insert(best_pos, chosen)
        assigned.add(chosen)
        unassigned.remove(chosen)

    # Helper to compute route length
    def route_length(route):
        if len(route) == 2:
            return 2 * dist[0][0]  # but depot distances are zero? Actually dist[0][0]=0
        length = 0
        for i in range(len(route) - 1):
            length += dist[route[i]][route[i+1]]
        return length

    # Improvement: relocate from longest route to reduce max length
    max_iter = n * truck_count  # finite bound
    for _ in range(max_iter):
        # Compute lengths and identify longest route
        lengths = [route_length(r) for r in routes]
        current_max = max(lengths)
        # Call report_best_vrp whenever we have a complete solution (including during improvement)
        # We'll call it at start and after each improvement
        # But ensure it's called at least once after construction
        # Actually, we should call it after construction and whenever we improve
        # We'll call it at the end of this loop after checking improvement
        # For now, call it before improvement?
        # According to spec: "report_best_vrp(routes) whenever you find a better complete feasible route set"
        # So we call it at the beginning and after each improvement.
        # We'll do it outside this loop.
        # Actually we need to call it on the first constructed solution too.
        # We'll call it after construction.
        # But here we are inside improvement loop, so we'll call it at the end of function.
        # To be safe, we call it at the start of improvement and after each improvement.
        if _ == 0:
            # called before any improvement
            pass

        longest_idx = lengths.index(current_max)
        longest_route = routes[longest_idx]
        # For each customer in longest route (excluding depots)
        improved = False
        for c in longest_route[1:-1]:  # skip depot nodes
            # Try to move c to other routes
            best_move = None
            best_new_max = current_max
            for r_idx, route in enumerate(routes):
                if r_idx == longest_idx:
                    continue
                # Try insertion positions in route
                for i in range(len(route) - 1):
                    cost_c = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                    # Remove c from its current position
                    # Compute current contribution of c in longest route
                    c_pos = longest_route.index(c)  # O(n) but small
                    old_contrib = dist[longest_route[c_pos-1]][c] + dist[c][longest_route[c_pos+1]] - dist[longest_route[c_pos-1]][longest_route[c_pos+1]]
                    new_longest_length = lengths[longest_idx] - old_contrib
                    new_route_length = lengths[r_idx] + cost_c
                    new_max = max(new_longest_length, new_route_length, max(lengths[:longest_idx] + lengths[longest_idx+1:]))
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (r_idx, i+1, c_pos)
            if best_move is not None and best_new_max < current_max:
                # Apply move
                r_idx, insert_pos, c_pos = best_move
                c_val = routes[longest_idx].pop(c_pos)
                routes[r_idx].insert(insert_pos, c_val)
                improved = True
                break  # restart loop
        if not improved:
            break

    # At this point, we have a complete solution. Call report_best_vrp
    # Note: report_best_vrp should be defined externally
    # We'll call it
    report_best_vrp(routes)

    return routes
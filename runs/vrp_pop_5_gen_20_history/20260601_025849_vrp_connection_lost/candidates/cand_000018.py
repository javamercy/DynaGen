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

    # Regret-insertion construction (identical to parent)
    while unassigned:
        best_costs = {}
        for c in unassigned:
            best = float('inf')
            second_best = float('inf')
            best_route = -1
            best_pos = -1
            for r_idx, route in enumerate(routes):
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

        regret = {}
        for c, (best, _, _, second) in best_costs.items():
            if second == float('inf'):
                regret[c] = float('inf')
            else:
                regret[c] = second - best

        max_regret = max(regret.values())
        candidates = [c for c in unassigned if regret[c] == max_regret]
        if len(candidates) > 1:
            candidates.sort(key=lambda c: (-best_costs[c][0], c))
        chosen = candidates[0]
        best_cost, best_route, best_pos, _ = best_costs[chosen]
        routes[best_route].insert(best_pos, chosen)
        assigned.add(chosen)
        unassigned.remove(chosen)

    # Helper to compute route length
    def route_length(route):
        if len(route) == 2:
            return 2 * dist[0][0]
        length = 0
        for i in range(len(route) - 1):
            length += dist[route[i]][route[i+1]]
        return length

    # Compute initial lengths and max
    lengths = [route_length(r) for r in routes]
    current_max = max(lengths)
    best_routes = [list(r) for r in routes]
    best_max = current_max

    # Call report_best_vrp on initial solution
    report_best_vrp(routes)

    # Simulated annealing improvement
    max_iter = n * truck_count * 10
    T = 1.0
    T_end = 0.0
    n_inner = n  # number of moves per temperature
    for iteration in range(max_iter):
        # Update temperature linearly
        T = 1.0 - (iteration / max_iter) * (1.0 - T_end)
        if T <= 0:
            break
        # For each inner loop, try a move
        for _ in range(n_inner):
            # Identify longest route
            max_len = max(lengths)
            longest_indices = [i for i, l in enumerate(lengths) if l == max_len]
            longest_idx = random.choice(longest_indices)
            longest_route = routes[longest_idx]
            if len(longest_route) <= 2:
                continue
            # Pick random customer from longest route (excluding depots)
            cust = random.choice(longest_route[1:-1])
            # Find its position
            c_pos = longest_route.index(cust)
            # Compute removal cost
            old_contrib = dist[longest_route[c_pos-1]][cust] + dist[cust][longest_route[c_pos+1]] - dist[longest_route[c_pos-1]][longest_route[c_pos+1]]
            new_longest_length = lengths[longest_idx] - old_contrib
            # Choose target route (different from longest_idx)
            target_idx = random.choice([r for r in range(truck_count) if r != longest_idx])
            target_route = routes[target_idx]
            # Choose random insertion position (after depot)
            insert_pos = random.randint(1, len(target_route) - 1)
            # Compute insertion cost
            cost_insert = dist[target_route[insert_pos-1]][cust] + dist[cust][target_route[insert_pos]] - dist[target_route[insert_pos-1]][target_route[insert_pos]]
            new_target_length = lengths[target_idx] + cost_insert
            # New max distance after move
            new_max = max(new_longest_length, new_target_length, max([l for i, l in enumerate(lengths) if i not in (longest_idx, target_idx)]))
            # Decide whether to accept
            delta = new_max - current_max
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T) if T > 0 else False
            if accept:
                # Apply move
                routes[longest_idx].pop(c_pos)
                routes[target_idx].insert(insert_pos, cust)
                lengths[longest_idx] = new_longest_length
                lengths[target_idx] = new_target_length
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(routes)
    # Return best found
    return best_routes
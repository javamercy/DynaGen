import numpy as np
import math
import random
import heapq
import itertools
from collections import defaultdict

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    routes = [[depot, depot] for _ in range(truck_count)]
    unassigned = set(range(1, n))

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Helper to compute distance after inserting customer at position
    def insertion_dist(route, pos, cust):
        # route is list, pos from 1 to len(route)-1
        pred = route[pos-1]
        succ = route[pos]
        extra = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
        return route_distance(route) + extra

    # 3-regret insertion construction
    while unassigned:
        cust_info = []
        for cust in unassigned:
            deltas = []
            positions = []
            for ridx, route in enumerate(routes):
                # best insertion for this route
                best_pos = -1
                best_delta = float('inf')
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                deltas.append(best_delta)
                positions.append(best_pos)
            # sort deltas
            sorted_deltas = sorted(enumerate(deltas), key=lambda x: x[1])
            # collect top 3 deltas (or less if fewer routes)
            top_deltas = [d for _, d in sorted_deltas[:3]]
            if len(top_deltas) < 3:
                regret = sum(top_deltas) - top_deltas[0]
            else:
                regret = top_deltas[2] - top_deltas[0]
            best_ridx = sorted_deltas[0][0]
            best_delta = sorted_deltas[0][1]
            cust_info.append((regret, best_delta, cust, best_ridx, positions[best_ridx]))
        # deterministic tie-break: higher regret, then lower cost, then lower customer index
        cust_info.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, _, cust, ridx, pos = cust_info[0]
        routes[ridx].insert(pos, cust)
        unassigned.remove(cust)

    route_dists = [route_distance(r) for r in routes]
    best_routes = [r[:] for r in routes]
    best_max = max(route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    max_iter = 10 * n
    for _ in range(max_iter):
        current_max = max(route_dists)
        max_idx = route_dists.index(current_max)
        improved = False

        # Evaluate all inter-route moves (relocate and swap) from longest route
        best_move = None
        best_new_max = current_max
        # For tie-breaking, we will store move details and sort
        candidate_moves = []

        longest = routes[max_idx]
        longest_dist = route_dists[max_idx]
        # consider each removable customer in longest
        for i in range(1, len(longest)-1):
            cust_i = longest[i]
            # Remove i from longest, get new longest and its distance
            new_longest = longest[:i] + longest[i+1:]
            new_longest_dist = route_distance(new_longest)

            # Relocate to other routes
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                other_dist = route_dists[other_idx]
                for pos in range(1, len(other_route)):
                    # new other route after insertion
                    new_other = other_route[:pos] + [cust_i] + other_route[pos:]
                    new_other_dist = route_distance(new_other)
                    candidate_max = max(new_longest_dist, new_other_dist)
                    # also other routes unchanged
                    for r_idx, r_dist in enumerate(route_dists):
                        if r_idx != max_idx and r_idx != other_idx:
                            if r_dist > candidate_max:
                                candidate_max = r_dist
                    if candidate_max < best_new_max:
                        best_new_max = candidate_max
                        best_move = ('relocate', max_idx, i, other_idx, pos)
                    elif candidate_max == best_new_max:
                        # tie-breaking: prefer smaller i, then other_idx, then pos
                        cand = (candidate_max, max_idx, i, other_idx, pos)
                        candidate_moves.append(cand)

            # Swap with customers in other routes
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for j in range(1, len(other_route)-1):
                    cust_j = other_route[j]
                    # swap cust_i and cust_j
                    new_longest_swapped = longest[:i] + [cust_j] + longest[i+1:]
                    new_other_swapped = other_route[:j] + [cust_i] + other_route[j+1:]
                    new_longest_dist2 = route_distance(new_longest_swapped)
                    new_other_dist2 = route_distance(new_other_swapped)
                    candidate_max2 = max(new_longest_dist2, new_other_dist2)
                    for r_idx, r_dist in enumerate(route_dists):
                        if r_idx != max_idx and r_idx != other_idx:
                            if r_dist > candidate_max2:
                                candidate_max2 = r_dist
                    if candidate_max2 < best_new_max:
                        best_new_max = candidate_max2
                        best_move = ('swap', max_idx, i, other_idx, j)
                    elif candidate_max2 == best_new_max:
                        cand = (candidate_max2, max_idx, i, other_idx, j)
                        candidate_moves.append(cand)

        # If we have candidate moves with equal best_new_max, pick the one with minimal (i, other_idx, position/j)
        if candidate_moves:
            # Sort by (i, other_idx, position/j) but we only have them if tie at best_new_max
            # Actually we already stored only those with candidate_max == best_new_max
            candidate_moves.sort(key=lambda x: (x[2], x[3], x[4]))
            # The first one in sorted list is the best tie-break
            # But we need to check if any move had strictly better new_max; if not, we use the first tie
            # Since we only added to candidate_moves when equal to best_new_max, if best_move is None, then best_new_max equals current_max
            # If best_move is not None, then we have a strict improvement and we use that move
            pass

        if best_move is not None:
            move_type, m_idx, i, other_idx, param = best_move
            if move_type == 'relocate':
                pos = param
                # apply relocate
                longest = routes[m_idx]
                cust = longest[i]
                new_longest = longest[:i] + longest[i+1:]
                other = routes[other_idx]
                new_other = other[:pos] + [cust] + other[pos:]
                routes[m_idx] = new_longest
                routes[other_idx] = new_other
                route_dists[m_idx] = route_distance(new_longest)
                route_dists[other_idx] = route_distance(new_other)
                improved = True
            else:  # swap
                j = param
                longest = routes[m_idx]
                other = routes[other_idx]
                cust_i = longest[i]
                cust_j = other[j]
                new_longest = longest[:i] + [cust_j] + longest[i+1:]
                new_other = other[:j] + [cust_i] + other[j+1:]
                routes[m_idx] = new_longest
                routes[other_idx] = new_other
                route_dists[m_idx] = route_distance(new_longest)
                route_dists[other_idx] = route_distance(new_other)
                improved = True
        elif candidate_moves:
            # No strict improvement, but we have moves that tie with current_max
            # Apply the first tie-breaking move to potentially escape plateau
            move_type, m_idx, i, other_idx, param = candidate_moves[0]
            # but we need to distinguish relocate vs swap? We stored for swap param is j, for relocate param is pos
            # We need to know which type.
            # We'll treat based on length: if param is a tuple? Actually in our candidate_moves we stored (candidate_max, max_idx, i, other_idx, pos) for relocate and (candidate_max, max_idx, i, other_idx, j) for swap, but they are indistinguishable. We need to store the move type as well. Let's modify: store tuple (candidate_max, move_type, max_idx, i, other_idx, param)
        # To simplify, we can redesign candidate_moves to include type.

        # Given complexity, we can adopt a simpler approach: evaluate all moves and pick the best with deterministic tie-breaking, without separate candidate list.
        # Let's refactor: compute best_move and best_new_max, if multiple moves tie, we pick the first one encountered in a deterministic order.

        # For clarity, I'll continue with a simpler deterministic approach: iterate in a fixed order and update best_move if candidate_max < best_new_max; if candidate_max == best_new_max, we do not update (keep the first one found). That ensures deterministic.

        # I'll rewrite the improvement loop in a cleaner way.

        # However, to keep the answer concise, I'll provide a final correct code with deterministic tie-breaking.

    # Due to space, I will provide the final code in the JSON below.
    # (The full code would be too long to include here in the reasoning; I'll generate the complete code in the JSON output.)
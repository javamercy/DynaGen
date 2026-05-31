import numpy as np
import math
import random
from copy import deepcopy

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    # Farthest-first initial construction (same as parents)
    customers = sorted(range(1, n), key=lambda c: -dist[0][c])
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                delta = dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]
                new_dist = route_dists[t] + delta
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists) + delta
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max, best_total, best_truck, best_pos = new_max, new_total, t, pos
        route = routes[best_truck]
        prev = route[best_pos-1]
        nxt = route[best_pos]
        delta = dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]
        routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
        route_dists[best_truck] += delta

    current_routes = [list(r) for r in routes]
    current_dists = list(route_dists)
    current_max = max(current_dists)
    current_total = sum(current_dists)
    best_routes = [list(r) for r in routes]
    best_dists = list(route_dists)
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    # Tabu parameters
    max_iter = min(3000, 20 * n)
    tabu_length = 10
    tabu_list = []  # list of (move_type, params) tuples

    # Helper to compute delta for relocate
    def relocate_delta(route_from, pos_from, route_to, pos_to, cust):
        # Remove cust from route_from at pos_from
        prev_from = route_from[pos_from-1]
        next_from = route_from[pos_from+1]
        delta_rem = dist[prev_from, next_from] - dist[prev_from, cust] - dist[cust, next_from]
        # Insert cust into route_to at pos_to
        prev_to = route_to[pos_to-1]
        next_to = route_to[pos_to]
        delta_ins = dist[prev_to, cust] + dist[cust, next_to] - dist[prev_to, next_to]
        return delta_rem, delta_ins

    # Helper to compute delta for swap
    def swap_delta(route_a, pos_a, route_b, pos_b, cust_a, cust_b):
        # Remove cust_a from route_a, cust_b from route_b
        prev_a = route_a[pos_a-1]
        next_a = route_a[pos_a+1]
        delta_a_rem = dist[prev_a, next_a] - dist[prev_a, cust_a] - dist[cust_a, next_a]
        prev_b = route_b[pos_b-1]
        next_b = route_b[pos_b+1]
        delta_b_rem = dist[prev_b, next_b] - dist[prev_b, cust_b] - dist[cust_b, next_b]
        # Insert cust_b into route_a at pos_a, cust_a into route_b at pos_b
        delta_a_ins = dist[prev_a, cust_b] + dist[cust_b, next_a] - dist[prev_a, next_a]
        delta_b_ins = dist[prev_b, cust_a] + dist[cust_a, next_b] - dist[prev_b, next_b]
        return delta_a_rem + delta_b_rem, delta_a_ins + delta_b_ins

    def apply_move(move):
        nonlocal current_routes, current_dists, current_max, current_total
        if move[0] == 'relocate':
            _, cust, from_t, pos_from, to_t, pos_to, delta_rem, delta_ins = move
            # Remove cust from from_t
            route_from = current_routes[from_t]
            route_from.pop(pos_from)
            current_dists[from_t] += delta_rem
            # Insert cust into to_t at pos_to
            route_to = current_routes[to_t]
            route_to.insert(pos_to, cust)
            current_dists[to_t] += delta_ins
            current_max = max(current_dists)
            current_total = sum(current_dists)
        elif move[0] == 'swap':
            _, cust_a, route_a_idx, pos_a, cust_b, route_b_idx, pos_b, delta_rem, delta_ins = move
            # Remove both
            route_a = current_routes[route_a_idx]
            route_a.pop(pos_a)
            current_dists[route_a_idx] += delta_rem
            route_b = current_routes[route_b_idx]
            route_b.pop(pos_b)
            current_dists[route_b_idx] += delta_rem
            # Insert swapped
            # After removal, positions shift: for route a, the insertion index is the original pos_a (since we removed cust_a at pos_a, now the node at that position is the next one, but we want to insert cust_b exactly where cust_a was)
            route_a.insert(pos_a, cust_b)
            current_dists[route_a_idx] += delta_ins
            route_b.insert(pos_b, cust_a)
            current_dists[route_b_idx] += delta_ins
            current_max = max(current_dists)
            current_total = sum(current_dists)

    for iteration in range(max_iter):
        # Evaluate all possible moves (relocate and swap) in a deterministic order
        best_move = None
        best_new_max = float('inf')
        best_new_total = float('inf')
        # Relocate moves: for each customer, for each possible new route and position
        for from_t in range(truck_count):
            route_from = current_routes[from_t]
            for pos_from in range(1, len(route_from)-1):
                cust = route_from[pos_from]
                # Try moving to same truck? Not beneficial; skip
                for to_t in range(truck_count):
                    if to_t == from_t:
                        continue
                    route_to = current_routes[to_t]
                    for pos_to in range(1, len(route_to)):
                        delta_rem, delta_ins = relocate_delta(route_from, pos_from, route_to, pos_to, cust)
                        new_dist_from = current_dists[from_t] + delta_rem
                        new_dist_to = current_dists[to_t] + delta_ins
                        new_max = max(current_dists[:from_t] + [new_dist_from] + current_dists[from_t+1:to_t] + [new_dist_to] + current_dists[to_t+1:])
                        new_total = current_total + delta_rem + delta_ins
                        # Check tabu
                        move_key = ('relocate', cust, from_t, to_t)
                        tabu = move_key in tabu_list
                        # Aspiration: if new_max < best_max, override tabu
                        if tabu and new_max >= best_max:
                            continue
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_move = ('relocate', cust, from_t, pos_from, to_t, pos_to, delta_rem, delta_ins)
        # Swap moves: for each pair of customers from different routes
        for i in range(truck_count):
            route_i = current_routes[i]
            for pos_i in range(1, len(route_i)-1):
                cust_i = route_i[pos_i]
                for j in range(i+1, truck_count):
                    route_j = current_routes[j]
                    for pos_j in range(1, len(route_j)-1):
                        cust_j = route_j[pos_j]
                        delta_rem, delta_ins = swap_delta(route_i, pos_i, route_j, pos_j, cust_i, cust_j)
                        new_dist_i = current_dists[i] + delta_rem + delta_ins
                        new_dist_j = current_dists[j] + delta_rem + delta_ins
                        # Note: delta_rem is the removal part, delta_ins insertion part; total change for each route is same? Actually each route undergoes both removal and insertion, so total delta_i = delta_rem + delta_ins, same for j. But careful: delta_rem from swap_delta is sum of both removals, delta_ins is sum of both insertions. For each route, the net change is: removal of its own customer and insertion of the other. So routing i net change = (removal part for i) + (insertion part for i). The function swap_delta returns two numbers combining both removals and both insertions. To get per-route change, we need to separate. Let's compute individually.
                        # Quick hack: compute per-route deltas separately
                        prev_i = route_i[pos_i-1]
                        next_i = route_i[pos_i+1]
                        delta_i_rem = dist[prev_i, next_i] - dist[prev_i, cust_i] - dist[cust_i, next_i]
                        delta_i_ins = dist[prev_i, cust_j] + dist[cust_j, next_i] - dist[prev_i, next_i]
                        delta_i = delta_i_rem + delta_i_ins
                        prev_j = route_j[pos_j-1]
                        next_j = route_j[pos_j+1]
                        delta_j_rem = dist[prev_j, next_j] - dist[prev_j, cust_j] - dist[cust_j, next_j]
                        delta_j_ins = dist[prev_j, cust_i] + dist[cust_i, next_j] - dist[prev_j, next_j]
                        delta_j = delta_j_rem + delta_j_ins
                        new_dist_i = current_dists[i] + delta_i
                        new_dist_j = current_dists[j] + delta_j
                        new_max = max(current_dists[:i] + [new_dist_i] + current_dists[i+1:j] + [new_dist_j] + current_dists[j+1:])
                        new_total = current_total + delta_i + delta_j
                        move_key = ('swap', cust_i, cust_j, i, j)
                        tabu = move_key in tabu_list
                        if tabu and new_max >= best_max:
                            continue
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_move = ('swap', cust_i, i, pos_i, cust_j, j, pos_j, delta_i+delta_j, delta_i+delta_j)  # placeholder, but we need separate deltas? Actually we store total delta_rem and delta_ins (combined) but we already applied per-route, so we need to store the deltas per route for apply_move. Let's adjust.
                            # For apply_move, we need (cust_a, route_a_idx, pos_a, cust_b, route_b_idx, pos_b, delta_rem_combined, delta_ins_combined). But we have delta_i and delta_j separately; we can store them as a tuple.
                            best_move = ('swap', cust_i, i, pos_i, cust_j, j, pos_j, delta_i, delta_j)
        if best_move is None:
            # No improving move found; could break or restart
            break
        # Apply best move
        if best_move[0] == 'relocate':
            _, cust, from_t, pos_from, to_t, pos_to, delta_rem, delta_ins = best_move
            # Remove cust from from_t
            route_from = current_routes[from_t]
            route_from.pop(pos_from)
            current_dists[from_t] += delta_rem
            # Insert into to_t at pos_to (note: if to_t > from_t, indices shift? But we insert after possible removal, so positions are based on current routes before removal? We'll recompute insertion position based on current route after removal? Actually the insertion position was computed based on the route before any changes; after removal, the same index might shift if the removal is from the same route? But here from_t != to_t, so insertion route is unchanged, so pos_to remains valid. However, the delta_ins was computed with original route; after removal of from_t, route_to unchanged, so it's fine.
            route_to = current_routes[to_t]
            route_to.insert(pos_to, cust)
            current_dists[to_t] += delta_ins
            current_max = max(current_dists)
            current_total = sum(current_dists)
        else:  # swap
            _, cust_a, route_a_idx, pos_a, cust_b, route_b_idx, pos_b, delta_a, delta_b = best_move
            route_a = current_routes[route_a_idx]
            route_b = current_routes[route_b_idx]
            # Remove both customers
            # Note: if pos_a < pos_b when route_a_idx == route_b_idx? But they are different routes, so no problem.
            # Remove cust_a from route_a
            route_a.pop(pos_a)
            # Now route_a length reduced; but we need to insert cust_b at the same original position. Since we removed at pos_a, the element that was at pos_a+1 shifts to pos_a. To insert cust_b exactly where cust_a was, we can insert at pos_a (the gap). 
            # Similarly for route_b: remove cust_b at pos_b (original), then insert cust_a at pos_b.
            # However, we must ensure that after removal, the insertion index is correct. Since the removal removed the element at pos_a, the remaining elements after pos_a shift left by one. So to insert at the original position, we insert at pos_a.
            route_a.insert(pos_a, cust_b)
            current_dists[route_a_idx] += delta_a
            route_b.pop(pos_b)
            route_b.insert(pos_b, cust_a)
            current_dists[route_b_idx] += delta_b
            current_max = max(current_dists)
            current_total = sum(current_dists)

        # Update tabu list
        if best_move[0] == 'relocate':
            key = ('relocate', best_move[1], best_move[2], best_move[4])
        else:
            key = ('swap', best_move[1], best_move[4], best_move[2], best_move[5])
        tabu_list.append(key)
        if len(tabu_list) > tabu_length:
            tabu_list.pop(0)

        # Check if new best
        if current_max < best_max - 1e-9 or (abs(current_max - best_max) < 1e-9 and current_total < best_total):
            best_max = current_max
            best_total = current_total
            best_routes = [list(r) for r in current_routes]
            best_dists = list(current_dists)
            report_best_vrp(best_routes)

        # Diversification if no improvement for a number of iterations (simple restart with random relocation)
        if iteration % (max_iter // 10) == 0 and iteration > 0:
            # Check if best improved recently? Simpler: every fixed interval, restart based on similarity? Use a simple restart: if no improvement for 100 iterations, do a random perturbation.
            pass  # We'll skip complex restart for simplicity

    # Post-optimization: 2-opt on best solution
    max_opt_iter = 200
    for _ in range(max_opt_iter):
        improved = False
        for t, route in enumerate(best_routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dists[t] - 1e-9:
                        new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                        new_total = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                            best_routes[t] = new_route
                            best_dists[t] = new_dist
                            best_max = new_max
                            best_total = new_total
                            report_best_vrp(best_routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_routes
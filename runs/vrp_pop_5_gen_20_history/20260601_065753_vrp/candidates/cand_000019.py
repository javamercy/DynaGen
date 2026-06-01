import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count

    def compute_route_length(route):
        length = 0.0
        for i in range(len(route) - 1):
            length += distance_matrix[route[i], route[i+1]]
        return length

    # Construction: min-max insertion (same as parent)
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            for p in range(1, len(route)):
                prev_node = route[p-1]
                next_node = route[p]
                old_edge = distance_matrix[prev_node, next_node]
                new_len = route_lengths[r] - old_edge + distance_matrix[prev_node, cust] + distance_matrix[cust, next_node]
                new_max = new_len
                for rr in range(truck_count):
                    if rr != r:
                        if route_lengths[rr] > new_max:
                            new_max = route_lengths[rr]
                if new_max < best_max:
                    best_max = new_max
                    best_route_idx = r
                    best_pos = p
                elif new_max == best_max:
                    # Tie-breaking: smaller route index, then smaller position
                    if r < best_route_idx or (r == best_route_idx and p < best_pos):
                        best_max = new_max
                        best_route_idx = r
                        best_pos = p
        # Insert customer
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_lengths[best_route_idx] = compute_route_length(route)

    current_max = max(route_lengths)
    best_max = current_max
    best_routes = [list(r) for r in routes]
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        max_cost = max(compute_route_length(r) for r in routes)
        if max_cost < best_max:
            best_max = max_cost
            best_routes = [list(r) for r in routes]
    report_best_vrp(routes)

    # Simulated annealing improvement
    T0 = current_max  # initial temperature
    T = T0
    alpha = 0.99
    max_iter = 50 * n
    for it in range(max_iter):
        # Choose a random move type
        move_type = random.choice(['relocate', 'swap', '2opt'])
        best_move = None
        best_delta = None
        if move_type == 'relocate':
            # Random relocate move
            t1 = random.randrange(truck_count)
            route1 = routes[t1]
            if len(route1) > 2:
                idx1 = random.randrange(1, len(route1)-1)
                cust = route1[idx1]
                t2 = random.randrange(truck_count)
                if t2 != t1:
                    route2 = routes[t2]
                    pos = random.randrange(1, len(route2))
                    # Compute delta
                    new_route1 = route1[:idx1] + route1[idx1+1:]
                    new_route2 = route2[:pos] + [cust] + route2[pos:]
                    len1_new = compute_route_length(new_route1)
                    len2_new = compute_route_length(new_route2)
                    new_max = max(len1_new, len2_new)
                    for rr in range(truck_count):
                        if rr != t1 and rr != t2:
                            if route_lengths[rr] > new_max:
                                new_max = route_lengths[rr]
                    delta = new_max - current_max
                    best_move = ('relocate', t1, idx1, t2, pos, new_route1, new_route2, len1_new, len2_new)
                    best_delta = delta
        elif move_type == 'swap':
            t1 = random.randrange(truck_count)
            route1 = routes[t1]
            if len(route1) > 2:
                idx1 = random.randrange(1, len(route1)-1)
                cust1 = route1[idx1]
                t2 = random.randrange(truck_count)
                if t2 != t1:
                    route2 = routes[t2]
                    if len(route2) > 2:
                        idx2 = random.randrange(1, len(route2)-1)
                        cust2 = route2[idx2]
                        new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                        new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                        len1_new = compute_route_length(new_route1)
                        len2_new = compute_route_length(new_route2)
                        new_max = max(len1_new, len2_new)
                        for rr in range(truck_count):
                            if rr != t1 and rr != t2:
                                if route_lengths[rr] > new_max:
                                    new_max = route_lengths[rr]
                        delta = new_max - current_max
                        best_move = ('swap', t1, idx1, t2, idx2, new_route1, new_route2, len1_new, len2_new)
                        best_delta = delta
        elif move_type == '2opt':
            t = random.randrange(truck_count)
            route = routes[t]
            if len(route) > 3:
                i = random.randrange(1, len(route)-2)
                j = random.randrange(i+1, len(route)-1)
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_len = compute_route_length(new_route)
                new_max = new_len
                for rr in range(truck_count):
                    if rr != t:
                        if route_lengths[rr] > new_max:
                            new_max = route_lengths[rr]
                delta = new_max - current_max
                best_move = ('2opt', t, i, j, new_route, new_len)
                best_delta = delta

        if best_move is not None:
            # Accept with probability
            if best_delta < 0 or random.random() < np.exp(-best_delta / T):
                # Apply move
                if best_move[0] == 'relocate':
                    _, t1, idx1, t2, pos, new_route1, new_route2, len1, len2 = best_move
                    routes[t1] = new_route1
                    routes[t2] = new_route2
                    route_lengths[t1] = len1
                    route_lengths[t2] = len2
                elif best_move[0] == 'swap':
                    _, t1, idx1, t2, idx2, new_route1, new_route2, len1, len2 = best_move
                    routes[t1] = new_route1
                    routes[t2] = new_route2
                    route_lengths[t1] = len1
                    route_lengths[t2] = len2
                elif best_move[0] == '2opt':
                    _, t, i, j, new_route, new_len = best_move
                    routes[t] = new_route
                    route_lengths[t] = new_len
                current_max = max(route_lengths)
                report_best_vrp(routes)
        T *= alpha

    return best_routes
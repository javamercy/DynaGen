import numpy as np

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

    # Construction: min-max insertion
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

    # Local search: only relocate moves (inter-route)
    max_iter = 2 * n
    for _ in range(max_iter):
        improved = False
        best_move = None
        best_new_max = current_max
        best_tie = None

        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust = route1[idx1]
                new_route1 = route1[:idx1] + route1[idx1+1:]
                len1_new = compute_route_length(new_route1)
                for t2 in range(truck_count):
                    if t1 == t2:
                        continue
                    route2 = routes[t2]
                    for pos in range(1, len(route2)):
                        new_route2 = route2[:pos] + [cust] + route2[pos:]
                        len2_new = compute_route_length(new_route2)
                        new_max = max(len1_new, len2_new)
                        for rr in range(truck_count):
                            if rr != t1 and rr != t2:
                                if route_lengths[rr] > new_max:
                                    new_max = route_lengths[rr]
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('relocate', t1, idx1, t2, pos)
                            best_tie = (t1, idx1, t2, pos)
                        elif new_max == best_new_max:
                            tie = (t1, idx1, t2, pos)
                            if best_tie is None or tie < best_tie:
                                best_new_max = new_max
                                best_move = ('relocate', t1, idx1, t2, pos)
                                best_tie = tie

        if best_move is not None and best_new_max < current_max:
            _, t1, idx1, t2, pos = best_move
            cust = routes[t1][idx1]
            del routes[t1][idx1]
            routes[t2].insert(pos, cust)
            # Update route lengths
            route_lengths = [compute_route_length(r) for r in routes]
            current_max = max(route_lengths)
            report_best_vrp(routes)
            improved = True
        if not improved:
            break

    return best_routes
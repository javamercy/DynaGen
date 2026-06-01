import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count

    def compute_route_length(route):
        length = 0.0
        for i in range(len(route) - 1):
            length += distance_matrix[route[i], route[i+1]]
        return length

    # Construction: greedy min-max insertion
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
                    if rr != r and route_lengths[rr] > new_max:
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
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_lengths[best_route_idx] = compute_route_length(route)

    def report_best_vrp(routes):
        pass

    current_max = max(route_lengths)
    report_best_vrp([list(r) for r in routes])

    # Local search: inter-route and intra-route relocate
    improved = True
    for iteration in range(2 * n):
        if not improved:
            break
        improved = False
        # Inter-route relocate
        for r_from in range(truck_count):
            route_from = routes[r_from]
            if len(route_from) <= 2:
                continue
            for idx in range(1, len(route_from)-1):
                cust = route_from[idx]
                # remove customer
                new_route_from = route_from[:idx] + route_from[idx+1:]
                len_from_new = compute_route_length(new_route_from)
                for r_to in range(truck_count):
                    if r_from == r_to:
                        continue
                    route_to = routes[r_to]
                    for p in range(1, len(route_to)):
                        new_route_to = route_to[:p] + [cust] + route_to[p:]
                        len_to_new = compute_route_length(new_route_to)
                        new_max = max(len_from_new, len_to_new)
                        for rr in range(truck_count):
                            if rr != r_from and rr != r_to:
                                if route_lengths[rr] > new_max:
                                    new_max = route_lengths[rr]
                        if new_max < current_max:
                            routes[r_from] = new_route_from
                            routes[r_to] = new_route_to
                            route_lengths[r_from] = len_from_new
                            route_lengths[r_to] = len_to_new
                            current_max = new_max
                            improved = True
                            report_best_vrp([list(r) for r in routes])
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Intra-route relocate
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 2:
                continue
            for idx in range(1, len(route)-1):
                cust = route[idx]
                new_route = route[:idx] + route[idx+1:]
                for p in range(1, len(new_route)):
                    new_route2 = new_route[:p] + [cust] + new_route[p:]
                    new_len = compute_route_length(new_route2)
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and route_lengths[rr] > new_max:
                            new_max = route_lengths[rr]
                    if new_max < current_max:
                        routes[r] = new_route2
                        route_lengths[r] = new_len
                        current_max = new_max
                        improved = True
                        report_best_vrp([list(r) for r in routes])
                        break
                if improved:
                    break
            if improved:
                break
    return [list(r) for r in routes]
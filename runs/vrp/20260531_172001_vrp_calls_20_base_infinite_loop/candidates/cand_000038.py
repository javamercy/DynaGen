import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    # Greedy min-max insertion
    unassigned = set(customers)
    while unassigned:
        best_customer = None
        best_route_idx = None
        best_pos = None
        best_max_after = float('inf')
        best_cost = None
        for cust in unassigned:
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    next_node = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, next_node] - distance_matrix[prev, next_node]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(r) for i, r in enumerate(routes) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if new_max < best_max_after or (new_max == best_max_after and (best_cost is None or cost > best_cost)):
                        best_max_after = new_max
                        best_customer = cust
                        best_route_idx = r_idx
                        best_pos = pos
                        best_cost = cost
                    elif new_max == best_max_after and cost == best_cost and cust < best_customer:
                        best_customer = cust
                        best_route_idx = r_idx
                        best_pos = pos
                        best_cost = cost
        route = routes[best_route_idx]
        route.insert(best_pos, best_customer)
        unassigned.remove(best_customer)

    # Improvement: limited iterations
    max_iter = n * truck_count
    for _ in range(max_iter):
        improved = False
        lengths = [route_length(r) for r in routes]
        current_max = max(lengths)
        # Inter-route relocate from longest route
        max_idx = np.argmax(lengths)
        max_route = routes[max_idx]
        if len(max_route) > 2:
            best_delta = 0
            best_move = None
            for cust in max_route[1:-1]:
                new_max_route = [x for x in max_route if x != cust]
                new_max_len = route_length(new_max_route)
                for r_idx in range(truck_count):
                    if r_idx == max_idx:
                        continue
                    other_route = routes[r_idx]
                    for pos in range(1, len(other_route)):
                        new_other_route = other_route[:pos] + [cust] + other_route[pos:]
                        new_other_len = route_length(new_other_route)
                        new_max_candidate = max(new_max_len, new_other_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)])
                        if new_max_candidate < current_max:
                            delta = current_max - new_max_candidate
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (cust, max_idx, r_idx, pos)
            if best_move:
                cust, from_idx, to_idx, pos = best_move
                routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                routes[to_idx].insert(pos, cust)
                improved = True
                report_best_vrp(routes)
        # Intra-route 2-opt (only if not improved yet, to avoid double work)
        if not improved:
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                best_gain = 0
                best_i = None
                best_k = None
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        gain = route_length(route) - (distance_matrix[route[i-1], route[i]] + distance_matrix[route[k], route[k+1]]) + (distance_matrix[route[i-1], route[k]] + distance_matrix[route[i], route[k+1]])
                        if gain > best_gain:
                            best_gain = gain
                            best_i = i
                            best_k = k
                if best_gain > 0:
                    route[best_i:best_k+1] = route[best_i:best_k+1][::-1]
                    improved = True
                    new_max = max(route_length(r) for r in routes)
                    if new_max < current_max:
                        report_best_vrp(routes)
        if not improved:
            break
    return routes
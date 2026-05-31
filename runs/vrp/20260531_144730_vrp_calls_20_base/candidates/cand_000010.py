import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unrouted = list(range(1, n))

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    def best_insertion(route, customer):
        best_pos = None
        best_delta = float('inf')
        for i in range(1, len(route)):
            delta = distance_matrix[route[i-1]][customer] + distance_matrix[customer][route[i]] - distance_matrix[route[i-1]][route[i]]
            if delta < best_delta:
                best_delta = delta
                best_pos = i
        return best_pos, best_delta

    # Phase 1: Cheapest insertion
    while unrouted:
        best_customer = None
        best_route_idx = None
        best_pos = None
        best_delta = float('inf')
        for cust in unrouted:
            for ridx, route in enumerate(routes):
                pos, delta = best_insertion(route, cust)
                if delta < best_delta:
                    best_delta = delta
                    best_customer = cust
                    best_route_idx = ridx
                    best_pos = pos
                elif delta == best_delta and cust < best_customer:
                    best_customer = cust
                    best_route_idx = ridx
                    best_pos = pos
        routes[best_route_idx].insert(best_pos, best_customer)
        unrouted.remove(best_customer)

    # Report initial solution
    report_best_vrp(routes)

    # Phase 2: Best-improvement local search to reduce max distance
    max_iter = n * truck_count
    for _ in range(max_iter):
        # Find current max distance route
        max_dist = 0
        max_route_idx = -1
        for idx, r in enumerate(routes):
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
                max_route_idx = idx

        best_move = None  # (cust_idx, customer, other_idx, pos, new_max)
        best_new_max = max_dist
        longest_route = routes[max_route_idx]
        for cust_idx in range(1, len(longest_route)-1):
            customer = longest_route[cust_idx]
            # Compute removal delta
            prev_node = longest_route[cust_idx-1]
            next_node = longest_route[cust_idx+1]
            removal_delta = distance_matrix[prev_node][next_node] - distance_matrix[prev_node][customer] - distance_matrix[customer][next_node]
            new_longest_dist = max_dist + removal_delta
            new_longest_route = longest_route[:cust_idx] + longest_route[cust_idx+1:]
            for other_idx in range(truck_count):
                if other_idx == max_route_idx:
                    continue
                other_route = routes[other_idx]
                pos, delta = best_insertion(other_route, customer)
                new_other_dist = route_distance(other_route) + delta
                potential_max = max(new_longest_dist, new_other_dist)
                # Check other routes unchanged
                for r_idx, r in enumerate(routes):
                    if r_idx != max_route_idx and r_idx != other_idx:
                        potential_max = max(potential_max, route_distance(r))
                if potential_max < best_new_max - 1e-12:
                    best_new_max = potential_max
                    best_move = (cust_idx, customer, other_idx, pos, new_longest_route)
                elif abs(potential_max - best_new_max) < 1e-12 and (best_move is None or customer < best_move[1]):
                    best_new_max = potential_max
                    best_move = (cust_idx, customer, other_idx, pos, new_longest_route)
        if best_move is not None:
            cust_idx, customer, other_idx, pos, new_longest_route = best_move
            routes[max_route_idx] = new_longest_route
            routes[other_idx].insert(pos, customer)
            report_best_vrp(routes)
        else:
            break
    # Ensure exactly truck_count routes
    for r in routes:
        if r[0] != 0 or r[-1] != 0:
            r.insert(0, 0)
            r.append(0)
    return routes
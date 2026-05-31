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

    report_best_vrp(routes)

    # Phase 2: Best-improvement move (customer from longest to another route)
    max_iter = n * truck_count
    for _ in range(max_iter):
        max_dist = 0
        max_route_idx = -1
        for idx, r in enumerate(routes):
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
                max_route_idx = idx

        best_move = None
        best_new_max = max_dist
        longest_route = routes[max_route_idx]
        for cust_idx in range(1, len(longest_route)-1):
            customer = longest_route[cust_idx]
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
                potential_max = new_longest_dist
                if new_other_dist > potential_max:
                    potential_max = new_other_dist
                for r_idx, r in enumerate(routes):
                    if r_idx != max_route_idx and r_idx != other_idx:
                        d = route_distance(r)
                        if d > potential_max:
                            potential_max = d
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

    # Phase 3: Best-improvement inter-route swap (exchange two customers between different routes)
    max_swap_iter = n * truck_count
    for _ in range(max_swap_iter):
        best_swap = None
        best_new_max_swap = float('inf')
        # Find current max distance
        current_max = 0
        for r in routes:
            d = route_distance(r)
            if d > current_max:
                current_max = d
        # Try all pairs of customers from different routes (depot excluded)
        for r_idx in range(truck_count):
            for i in range(1, len(routes[r_idx])-1):
                cust_i = routes[r_idx][i]
                for s_idx in range(r_idx+1, truck_count):
                    for j in range(1, len(routes[s_idx])-1):
                        cust_j = routes[s_idx][j]
                        # Compute new routes if swap
                        new_route_r = routes[r_idx][:i] + [cust_j] + routes[r_idx][i+1:]
                        new_route_s = routes[s_idx][:j] + [cust_i] + routes[s_idx][j+1:]
                        new_dist_r = route_distance(new_route_r)
                        new_dist_s = route_distance(new_route_s)
                        # Compute max of all routes
                        new_max = new_dist_r
                        if new_dist_s > new_max:
                            new_max = new_dist_s
                        for t_idx, rt in enumerate(routes):
                            if t_idx != r_idx and t_idx != s_idx:
                                d = route_distance(rt)
                                if d > new_max:
                                    new_max = d
                        if new_max < best_new_max_swap - 1e-12:
                            best_new_max_swap = new_max
                            best_swap = (r_idx, i, cust_i, s_idx, j, cust_j, new_route_r, new_route_s)
                        elif abs(new_max - best_new_max_swap) < 1e-12:
                            # tie-breaking: smaller customer id (cust_i or cust_j? use pair min)
                            if best_swap is None:
                                best_swap = (r_idx, i, cust_i, s_idx, j, cust_j, new_route_r, new_route_s)
                            else:
                                # if current best has smaller cust_i or cust_j, keep; else replace? Use lexicographic on (cust_i, cust_j)
                                curr_cust_i = best_swap[2]
                                curr_cust_j = best_swap[5]
                                if (cust_i < curr_cust_i) or (cust_i == curr_cust_i and cust_j < curr_cust_j):
                                    best_swap = (r_idx, i, cust_i, s_idx, j, cust_j, new_route_r, new_route_s)
        if best_swap is not None and best_new_max_swap < current_max - 1e-12:
            r_idx, i, _, s_idx, j, _, new_route_r, new_route_s = best_swap
            routes[r_idx] = new_route_r
            routes[s_idx] = new_route_s
            report_best_vrp(routes)
        else:
            break

    # Ensure exactly truck_count routes with start/end 0 (already)
    for r in routes:
        if r[0] != 0 or r[-1] != 0:
            r.insert(0, 0)
            r.append(0)
    return routes
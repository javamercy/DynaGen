import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    customers.sort(key=lambda c: distance_matrix[0, c], reverse=True)

    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count

    best_max = float('inf')
    best_routes = None

    def route_length(route):
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def update_best():
        nonlocal best_max, best_routes
        max_len = max(route_lengths)
        if max_len < best_max:
            best_max = max_len
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    def best_insertion(route, route_len, customer):
        best_pos = -1
        best_increase = float('inf')
        for i in range(1, len(route)):
            prev = route[i-1]
            nxt = route[i]
            increase = distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]
            if increase < best_increase - 1e-12:
                best_increase = increase
                best_pos = i
        return best_pos, best_increase

    # Construction
    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_new_max = float('inf')
        best_len_sum = float('inf')  # for tie-breaking
        for r_idx in range(truck_count):
            route = routes[r_idx]
            route_len = route_lengths[r_idx]
            pos, inc = best_insertion(route, route_len, cust)
            new_len = route_len + inc
            other_max = max(route_lengths[:r_idx] + route_lengths[r_idx+1:]) if truck_count > 1 else 0
            new_max = max(other_max, new_len)
            if new_max < best_new_max - 1e-12:
                best_new_max = new_max
                best_route_idx = r_idx
                best_pos = pos
                best_len_sum = new_len  # just store for tie
            elif abs(new_max - best_new_max) < 1e-12:
                # tie: choose route with smaller current length (to balance)
                if route_len < route_lengths[best_route_idx]:
                    best_route_idx = r_idx
                    best_pos = pos
                    best_len_sum = new_len
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_lengths[best_route_idx] = route_length(route)
    update_best()

    # Intra-route 2-opt on all routes
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        max_iter = len(route) * len(route)
        for _ in range(max_iter):
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
            if not improved:
                break
        route_lengths[r_idx] = route_length(route)
    update_best()

    # Main improvement loop: repeat up to n times
    for _ in range(n):
        # Find longest route
        max_len = max(route_lengths)
        max_idx = route_lengths.index(max_len)
        improved = False

        # Best-improvement relocation: try all relocations that reduce max
        best_move = None
        best_new_max = max_len  # initial
        for c in customers:
            # find current route of c
            for r_idx, route in enumerate(routes):
                if c in route:
                    break
            old_route = route
            old_dist = route_lengths[r_idx]
            new_route = [x for x in old_route if x != c]
            new_dist = route_length(new_route)
            for r2_idx, r2 in enumerate(routes):
                if r2_idx == r_idx:
                    continue
                for pos in range(1, len(r2)):
                    new_dist2 = route_lengths[r2_idx] - distance_matrix[r2[pos-1], r2[pos]] + distance_matrix[r2[pos-1], c] + distance_matrix[c, r2[pos]]
                    new_max = max(route_lengths[:r_idx] + [new_dist] + route_lengths[r_idx+1:r2_idx] + [new_dist2] + route_lengths[r2_idx+1:])
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_move = (r_idx, r2_idx, c, pos, new_route, new_dist, new_dist2)
        if best_move is not None:
            r_idx, r2_idx, c, pos, new_route, new_dist, new_dist2 = best_move
            routes[r_idx] = new_route
            routes[r2_idx] = routes[r2_idx][:pos] + [c] + routes[r2_idx][pos:]
            route_lengths[r_idx] = new_dist
            route_lengths[r2_idx] = new_dist2
            update_best()
            improved = True

        if not improved:
            # Best-improvement exchange: swap two customers from different routes
            best_move = None
            best_new_max = max_len
            for i in range(truck_count):
                for j in range(i+1, truck_count):
                    route_i = routes[i]
                    route_j = routes[j]
                    len_i = route_lengths[i]
                    len_j = route_lengths[j]
                    for ci in route_i[1:-1]:
                        for cj in route_j[1:-1]:
                            # remove ci from i, cj from j
                            new_route_i = [x for x in route_i if x != ci]
                            new_route_j = [x for x in route_j if x != cj]
                            # best insertion of ci into j, cj into i
                            best_pos_i, inc_i = best_insertion(new_route_i, route_length(new_route_i), cj)
                            best_pos_j, inc_j = best_insertion(new_route_j, route_length(new_route_j), ci)
                            new_len_i = route_length(new_route_i) + inc_i
                            new_len_j = route_length(new_route_j) + inc_j
                            new_max = max(route_lengths[:i] + [new_len_i] + route_lengths[i+1:j] + [new_len_j] + route_lengths[j+1:])
                            if new_max < best_new_max - 1e-12:
                                best_new_max = new_max
                                best_move = (i, j, ci, cj, new_route_i, new_route_j, best_pos_i, best_pos_j)
            if best_move is not None:
                i, j, ci, cj, new_route_i, new_route_j, pos_i, pos_j = best_move
                routes[i] = new_route_i[:pos_i] + [cj] + new_route_i[pos_i:]
                routes[j] = new_route_j[:pos_j] + [ci] + new_route_j[pos_j:]
                route_lengths[i] = route_length(routes[i])
                route_lengths[j] = route_length(routes[j])
                update_best()
                improved = True

        if not improved:
            break  # no improvement found

    if best_routes is None:
        best_routes = routes
    return best_routes
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

    def best_insertion(route, route_len, customer):
        best_pos = -1
        best_increase = float('inf')
        for i in range(1, len(route)):
            prev = route[i-1]
            nxt = route[i]
            increase = distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]
            if increase < best_increase:
                best_increase = increase
                best_pos = i
        return best_pos, best_increase

    # Construction
    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_new_max = float('inf')
        for r_idx in range(truck_count):
            route = routes[r_idx]
            route_len = route_lengths[r_idx]
            pos, inc = best_insertion(route, route_len, cust)
            new_len = route_len + inc
            other_max = max(route_lengths[:r_idx] + route_lengths[r_idx+1:]) if truck_count > 1 else 0
            new_max = max(other_max, new_len)
            if new_max < best_new_max or (new_max == best_new_max and r_idx < best_route_idx):
                best_new_max = new_max
                best_route_idx = r_idx
                best_pos = pos
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_lengths[best_route_idx] = route_length(route)
    update_best()

    # Intra-route 2-opt
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

    # Inter-route relocation
    improved = True
    max_iters = n * n
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for c in customers:
            # find current route of c
            for r_idx, route in enumerate(routes):
                if c in route:
                    break
            old_route = route
            old_dist = route_lengths[r_idx]
            new_route = [x for x in old_route if x != c]
            new_dist = route_length(new_route)
            # try inserting into other routes
            for r2_idx, r2 in enumerate(routes):
                if r2_idx == r_idx:
                    continue
                for pos in range(1, len(r2)):
                    new_dist2 = route_lengths[r2_idx] - distance_matrix[r2[pos-1], r2[pos]] + distance_matrix[r2[pos-1], c] + distance_matrix[c, r2[pos]]
                    new_max_val = max(route_lengths[:r_idx] + [new_dist] + route_lengths[r_idx+1:r2_idx] + [new_dist2] + route_lengths[r2_idx+1:])
                    if new_max_val < best_max:
                        # apply move
                        routes[r_idx] = new_route
                        routes[r2_idx] = r2[:pos] + [c] + r2[pos:]
                        route_lengths[r_idx] = new_dist
                        route_lengths[r2_idx] = new_dist2
                        best_max = new_max_val
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    if best_routes is None:
        best_routes = routes
    return best_routes
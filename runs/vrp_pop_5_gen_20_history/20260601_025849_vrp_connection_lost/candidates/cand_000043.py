import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)

    def insert_customer(route, pos, cust):
        return route[:pos] + [cust] + route[pos:]

    # Regret insertion
    while unassigned:
        best_cust = None
        best_regret = -1.0
        best_route_idx = -1
        best_pos = -1
        # For each customer, compute best and second-best insertion cost
        for cust in sorted(unassigned):
            best_cost = float('inf')
            second_best_cost = float('inf')
            best_r = -1
            best_p = -1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = insert_customer(route, pos, cust)
                    new_dist = route_distance(new_route)
                    other_max = max((route_distance(routes[i]) for i in range(truck_count) if i != r_idx), default=0.0)
                    new_max = max(new_dist, other_max)
                    if new_max < best_cost:
                        second_best_cost = best_cost
                        best_cost = new_max
                        best_r = r_idx
                        best_p = pos
                    elif new_max < second_best_cost:
                        second_best_cost = new_max
            if best_cost == float('inf'):
                continue
            regret = second_best_cost - best_cost
            # Deterministic tie-break: higher regret, then lower customer index
            if regret > best_regret or (regret == best_regret and cust < best_cust):
                best_regret = regret
                best_cust = cust
                best_route_idx = best_r
                best_pos = best_p
        # Insert best customer
        if best_cust is None:
            break
        route = routes[best_route_idx]
        routes[best_route_idx] = insert_customer(route, best_pos, best_cust)
        unassigned.remove(best_cust)

    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)

    # Local search improvement (same as parent)
    improved = True
    max_iter = n * n
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        # 2-opt for each route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        routes[r_idx] = new_route
                        improved = True
                        current_max = max_route_distance(routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                        break
                if improved:
                    break
        if improved:
            continue
        # Relocate: try moving from longest route to others
        max_dist = max_route_distance(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_distance(r) == max_dist]
        for r_idx in longest_indices:
            if len(routes[r_idx]) <= 3:
                continue
            for pos in range(1, len(routes[r_idx])-1):
                cust = routes[r_idx][pos]
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        continue
                    for other_pos in range(1, len(other_route)):
                        new_other = insert_customer(other_route, other_pos, cust)
                        new_self = routes[r_idx][:pos] + routes[r_idx][pos+1:]
                        new_routes = list(routes)
                        new_routes[r_idx] = new_self
                        new_routes[other_idx] = new_other
                        new_max = max_route_distance(new_routes)
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in new_routes]
                            routes = new_routes
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_routes
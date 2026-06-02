import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count <= 0:
        return []
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        routes += [[0, 0]] * (truck_count - len(routes))
        return routes

    # route distance helper
    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # regret insertion heuristic: insert a list of customers into current routes
    def regret_insert(unassigned, routes, route_dists):
        max_dist = max(route_dists) if routes else 0.0
        inserted = set()
        while unassigned:
            best_cust = None
            best_regret = -1.0
            # for each customer, compute best and second best insertion
            for cust in unassigned:
                best_inc = float('inf')
                second_inc = float('inf')
                best_route = -1
                best_pos = -1
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                        new_route_dist = route_dists[r_idx] + inc
                        # compute new max distance
                        other_max = max(route_dists[:r_idx] + route_dists[r_idx+1:]) if truck_count > 1 else 0.0
                        new_max = max(new_route_dist, other_max)
                        # determine if this is best or second best
                        if new_max < best_inc - 1e-12:
                            second_inc = best_inc
                            best_inc = new_max
                            best_route = r_idx
                            best_pos = pos
                        elif new_max < second_inc - 1e-12:
                            second_inc = new_max
                regret = second_inc - best_inc if second_inc < float('inf') else best_inc
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_route_sel = best_route
                    best_pos_sel = best_pos
                elif abs(regret - best_regret) < 1e-12 and cust < best_cust:
                    best_cust = cust
                    best_route_sel = best_route
                    best_pos_sel = best_pos
            # insert best_cust
            route = routes[best_route_sel]
            pos = best_pos_sel
            prev = route[pos-1]
            nxt = route[pos]
            inc = distance_matrix[prev][best_cust] + distance_matrix[best_cust][nxt] - distance_matrix[prev][nxt]
            route_dists[best_route_sel] += inc
            route.insert(pos, best_cust)
            unassigned.remove(best_cust)
            inserted.add(best_cust)

    # initial construction: routes with depot only
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = set(customers)
    regret_insert(unassigned, routes, route_dists)
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)

    # threshold accepting parameters
    max_iter = min(n * 10, 500)
    initial_threshold = 0.1 * best_max
    if initial_threshold < 1e-9:
        initial_threshold = 1.0
    threshold = initial_threshold

    for _ in range(max_iter):
        # choose the route with maximum distance (smallest index if tie)
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        destroy_route_idx = min(max_routes)
        # get customers on that route (excluding depots)
        destroy_route = routes[destroy_route_idx]
        if len(destroy_route) <= 2:
            continue
        removed_customers = destroy_route[1:-1]
        # save current state
        old_routes = [list(r) for r in routes]
        old_route_dists = list(route_dists)
        # remove customers
        routes[destroy_route_idx] = [0, 0]
        route_dists[destroy_route_idx] = 0.0
        # reinsert removed customers in deterministic order (ascending index)
        removed_sorted = sorted(removed_customers)
        regret_insert(removed_sorted, routes, route_dists)  # modifies sets, but we pass list; function expects set? actually we pass list and it removes items, but since we pass a list we need to convert to set? The function expects a set/mutable collection; we'll pass a list and it will call remove on it. It's okay because we pass a list copy.
        # compute new max
        new_max = max(route_dists)
        # check acceptance
        if new_max <= best_max + threshold:
            # accept the new solution
            if new_max < best_max - 1e-12:
                best_routes = [list(r) for r in routes]
                best_max = new_max
                report_best_vrp(best_routes)
        else:
            # revert
            routes = old_routes
            route_dists = old_route_dists
        # decrease threshold
        threshold = initial_threshold * (1 - (_+1)/max_iter)
        if threshold < 0:
            threshold = 0.0

    return best_routes
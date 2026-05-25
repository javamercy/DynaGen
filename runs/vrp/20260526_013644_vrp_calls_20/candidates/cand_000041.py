import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(cust, route):
        best_cost = float('inf')
        best_pos = -1
        for pos in range(1, len(route)):
            i = route[pos-1]
            j = route[pos]
            cost = distance_matrix[i, cust] + distance_matrix[cust, j] - distance_matrix[i, j]
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        return best_cost, best_pos

    def construct_routes():
        routes = [[depot, depot] for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            best_regret = -1.0
            best_cust = -1
            best_route_idx = -1
            best_pos = -1
            best_cost_for_cust = float('inf')
            for cust in list(unassigned):
                costs = []
                for r_idx, route in enumerate(routes):
                    cost, pos = best_insertion(cust, route)
                    costs.append((cost, r_idx, pos))
                costs.sort(key=lambda x: x[0])
                if len(costs) == 1:
                    regret = 1e9
                else:
                    regret = costs[1][0] - costs[0][0]
                if (regret > best_regret or
                    (regret == best_regret and costs[0][0] > best_cost_for_cust) or
                    (regret == best_regret and costs[0][0] == best_cost_for_cust and cust < best_cust)):
                    best_regret = regret
                    best_cust = cust
                    best_cost_for_cust = costs[0][0]
                    best_route_idx = costs[0][1]
                    best_pos = costs[0][2]
            route = routes[best_route_idx]
            route.insert(best_pos, best_cust)
            unassigned.remove(best_cust)
        return routes

    routes = construct_routes()
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)

    n_cust = n - 1
    max_iters_per_restart = min(300, 3 * n_cust)
    max_stall = 20
    max_perturbs = 2
    max_restarts = 2

    for restart in range(max_restarts + 1):
        if restart > 0:
            routes = construct_routes()
            current_max = max(route_dist(r) for r in routes)
            if current_max < best_max:
                best_routes = [list(r) for r in routes]
                best_max = current_max
                report_best_vrp(best_routes)

        perturb_count = 0
        stall_count = 0
        threshold = 0.1 * best_max if best_max > 0 else 0.0

        while perturb_count < max_perturbs:
            improved = False
            for it in range(max_iters_per_restart):
                # Inter-route relocate (best improvement on max with threshold)
                best_move = None
                best_new_max = best_max
                for r_idx, route in enumerate(routes):
                    if len(route) <= 3:
                        continue
                    for cust in route[1:-1]:
                        new_route = [x for x in route if x != cust]
                        for other_idx, other_route in enumerate(routes):
                            if other_idx == r_idx:
                                continue
                            cost, pos = best_insertion(cust, other_route)
                            candidate_routes = [list(r) for r in routes]
                            candidate_routes[r_idx] = new_route
                            other_new = list(other_route)
                            other_new.insert(pos, cust)
                            candidate_routes[other_idx] = other_new
                            dists = [route_dist(r) for r in candidate_routes]
                            new_max = max(dists)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = candidate_routes
                if best_move is not None and best_new_max <= best_max + threshold:
                    routes = best_move
                    if best_new_max < best_max:
                        best_max = best_new_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                        threshold = 0.1 * best_max
                    improved = True
                    continue

                # Cross-route 2-opt* (best improvement on max with threshold)
                best_move = None
                best_new_max = best_max
                for r1_idx in range(truck_count):
                    r1 = routes[r1_idx]
                    if len(r1) <= 3:
                        continue
                    for i in range(1, len(r1)-1):
                        for r2_idx in range(r1_idx+1, truck_count):
                            r2 = routes[r2_idx]
                            if len(r2) <= 3:
                                continue
                            for j in range(1, len(r2)-1):
                                new_r1 = r1[:i+1] + r2[j+1:]
                                new_r2 = r2[:j+1] + r1[i+1:]
                                candidate_routes = [list(r) for r in routes]
                                candidate_routes[r1_idx] = new_r1
                                candidate_routes[r2_idx] = new_r2
                                dists = [route_dist(r) for r in candidate_routes]
                                new_max = max(dists)
                                if new_max < best_new_max:
                                    best_new_max = new_max
                                    best_move = candidate_routes
                if best_move is not None and best_new_max <= best_max + threshold:
                    routes = best_move
                    if best_new_max < best_max:
                        best_max = best_new_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                        threshold = 0.1 * best_max
                    improved = True
                    continue

                # Intra-route 2-opt (best improvement on each route's own distance, no threshold needed for individual route)
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 4:
                        continue
                    best_dist = route_dist(route)
                    best_new_route = None
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_dist = route_dist(new_route)
                            if new_dist < best_dist:
                                best_dist = new_dist
                                best_new_route = new_route
                    if best_new_route is not None:
                        routes[r_idx] = best_new_route
                        improved = True
                        dists = [route_dist(r) for r in routes]
                        new_max = max(dists)
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            threshold = 0.1 * best_max

                if improved:
                    break

            if not improved:
                stall_count += 1
                if stall_count >= max_stall:
                    # Perturbation: destroy worst route and reinsert via regret
                    worst_idx = max(range(truck_count), key=lambda idx: (route_dist(routes[idx]), -idx))
                    worst_route = routes[worst_idx]
                    if len(worst_route) > 2:
                        customers_to_reinsert = worst_route[1:-1]
                        routes[worst_idx] = [depot, depot]
                        unassigned = set(customers_to_reinsert)
                        while unassigned:
                            best_regret = -1.0
                            best_cust = -1
                            best_route_idx = -1
                            best_pos = -1
                            best_cost_for_cust = float('inf')
                            for cust in list(unassigned):
                                costs = []
                                for r_idx, route in enumerate(routes):
                                    cost, pos = best_insertion(cust, route)
                                    costs.append((cost, r_idx, pos))
                                costs.sort(key=lambda x: x[0])
                                if len(costs) == 1:
                                    regret = 1e9
                                else:
                                    regret = costs[1][0] - costs[0][0]
                                if (regret > best_regret or
                                    (regret == best_regret and costs[0][0] > best_cost_for_cust) or
                                    (regret == best_regret and costs[0][0] == best_cost_for_cust and cust < best_cust)):
                                    best_regret = regret
                                    best_cust = cust
                                    best_cost_for_cust = costs[0][0]
                                    best_route_idx = costs[0][1]
                                    best_pos = costs[0][2]
                            route = routes[best_route_idx]
                            route.insert(best_pos, best_cust)
                            unassigned.remove(best_cust)
                    perturb_count += 1
                    stall_count = 0
                    current_max = max(route_dist(r) for r in routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                else:
                    break
            else:
                stall_count = 0
                perturb_count = 0

    # Ensure exactly truck_count routes
    result = []
    for r in best_routes:
        if len(r) <= 2:
            result.append([0, 0])
        else:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
            result.append(r)
    while len(result) < truck_count:
        result.append([0, 0])
    return result
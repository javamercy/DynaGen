import math
import heapq

import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]  # number of nodes (including depot 0)
    depot = 0
    customers = list(range(1, n))
    unassigned = set(customers)

    # Initialize empty routes
    routes = [[depot, depot] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count

    # Helper functions
    def route_dist(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i + 1]]
        return d

    def update_route_dists():
        for i, r in enumerate(routes):
            route_dists[i] = route_dist(r)
    update_route_dists()

    # Helper to compute new max distance if customer cust inserted into route r_idx at position pos
    def new_max_if_insert(cust, r_idx, pos):
        route = routes[r_idx]
        prev = route[pos - 1]
        next_ = route[pos]
        delta = distance_matrix[prev, cust] + distance_matrix[cust, next_] - distance_matrix[prev, next_]
        new_dist = route_dists[r_idx] + delta
        # Compute new max among all routes
        candidate_max = new_dist
        for i in range(truck_count):
            if i == r_idx:
                continue
            if route_dists[i] > candidate_max:
                candidate_max = route_dists[i]
        return candidate_max, new_dist

    # Regret-based construction
    while unassigned:
        best_regret = -1e9
        best_cust = -1
        best_r_idx = -1
        best_pos = -1
        best_new_dist = -1.0
        best_cost_for_cust = None

        for cust in unassigned:
            costs = []
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_max, new_dist = new_max_if_insert(cust, r_idx, pos)
                    costs.append((new_max, r_idx, pos, new_dist))
            # sort by new_max ascending
            costs.sort(key=lambda x: x[0])
            if len(costs) == 1:
                regret = costs[0][0] * 2  # large regret
                best_candidate = costs[0]
            else:
                best = costs[0]
                second_best = costs[1]
                regret = second_best[0] - best[0]
                best_candidate = best
            tie_breaker = best_candidate[0]  # best cost
            if regret > best_regret or (regret == best_regret and (best_cost_for_cust is None or tie_breaker > best_cost_for_cust)):
                best_regret = regret
                best_cust = cust
                best_r_idx = best_candidate[1]
                best_pos = best_candidate[2]
                best_new_dist = best_candidate[3]
                best_cost_for_cust = tie_breaker
            elif regret == best_regret and tie_breaker == best_cost_for_cust:
                if cust < best_cust:
                    best_cust = cust
                    best_r_idx = best_candidate[1]
                    best_pos = best_candidate[2]
                    best_new_dist = best_candidate[3]

        routes[best_r_idx].insert(best_pos, best_cust)
        route_dists[best_r_idx] = best_new_dist
        unassigned.remove(best_cust)

    # Post-construction: compute current best
    current_max = max(route_dists)
    best_routes = [list(r) for r in routes]
    best_max = current_max
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # Local search improvement
    max_iter = (n - 1) * 2
    for _ in range(max_iter):
        improved = False

        # Inter-route relocate
        for r_from in range(truck_count):
            if len(routes[r_from]) <= 2:
                continue
            cust_positions = list(range(1, len(routes[r_from]) - 1))
            for pos_from in cust_positions:
                cust = routes[r_from][pos_from]
                # Remove cust from r_from
                new_route_from = routes[r_from][:pos_from] + routes[r_from][pos_from + 1:]
                new_dist_from = route_dist(new_route_from)
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = routes[r_to]
                    for pos_to in range(1, len(route_to)):
                        new_route_to = route_to[:pos_to] + [cust] + route_to[pos_to:]
                        new_dist_to = route_dist(new_route_to)
                        cand_max = max(new_dist_from, new_dist_to)
                        for r in range(truck_count):
                            if r != r_from and r != r_to:
                                cand_max = max(cand_max, route_dists[r])
                        if cand_max < best_max:
                            best_max = cand_max
                            best_routes = [list(r) for r in routes]
                            best_routes[r_from] = new_route_from
                            best_routes[r_to] = new_route_to
                            routes[r_from] = new_route_from
                            routes[r_to] = new_route_to
                            route_dists[r_from] = new_dist_from
                            route_dists[r_to] = new_dist_to
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Inter-route swap
        for r1 in range(truck_count):
            if len(routes[r1]) <= 2:
                continue
            for pos1 in range(1, len(routes[r1]) - 1):
                cust1 = routes[r1][pos1]
                for r2 in range(r1 + 1, truck_count):
                    if len(routes[r2]) <= 2:
                        continue
                    for pos2 in range(1, len(routes[r2]) - 1):
                        cust2 = routes[r2][pos2]
                        new_route1 = list(routes[r1])
                        new_route2 = list(routes[r2])
                        new_route1[pos1] = cust2
                        new_route2[pos2] = cust1
                        new_dist1 = route_dist(new_route1)
                        new_dist2 = route_dist(new_route2)
                        cand_max = max(new_dist1, new_dist2)
                        for r in range(truck_count):
                            if r != r1 and r != r2:
                                cand_max = max(cand_max, route_dists[r])
                        if cand_max < best_max:
                            best_max = cand_max
                            best_routes = [list(r) for r in routes]
                            best_routes[r1] = new_route1
                            best_routes[r2] = new_route2
                            routes[r1] = new_route1
                            routes[r2] = new_route2
                            route_dists[r1] = new_dist1
                            route_dists[r2] = new_dist2
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 4:
                continue
            best_local_imp = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_dists[r_idx]:
                        cand_max = max(new_dist, max(route_dists[:r_idx] + route_dists[r_idx + 1:]))
                        if cand_max < best_max:
                            best_max = cand_max
                            best_routes = [list(r) for r in routes]
                            best_routes[r_idx] = new_route
                            routes[r_idx] = new_route
                            route_dists[r_idx] = new_dist
                            improved = True
                            best_local_imp = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                if best_local_imp:
                    break
            if improved:
                break
        if improved:
            continue

        # If no improvement, perform deterministic shake
        # Remove a fraction of customers (10% of unassigned) and reinsert using regret
        num_customers = n - 1
        shake_size = max(1, num_customers // 10)
        # Identify customers to remove: those with highest second-best cost (from regret) or something deterministic
        # We'll compute for each customer in routes, the cost if we remove them
        shake_candidates = []
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 2:
                continue
            for pos in range(1, len(route) - 1):
                cust = route[pos]
                # compute removal cost: increase in max route distance if removed?
                # Actually we want to remove customers that are likely to be in problematic positions.
                # Simple: remove customers from the route with maximum distance
                # To be deterministic, we'll sort by some measure
                # Here, we compute the marginal increase in max distance if this customer is inserted elsewhere
                # But easier: just remove customers from the longest route first
                pass
        # For simplicity, we remove customers from the route with the maximum distance
        max_route_idx = max(range(truck_count), key=lambda i: route_dists[i])
        route_to_shake = routes[max_route_idx]
        if len(route_to_shake) > 2:
            # remove up to shake_size customers from this route (excluding depots)
            customers_in_route = route_to_shake[1:-1]
            # sort by distance contribution? not necessary, just remove first shake_size
            remove_set = set(customers_in_route[:shake_size])
            for cust in remove_set:
                # remove from route
                idx = routes[max_route_idx].index(cust)
                routes[max_route_idx].pop(idx)
                route_dists[max_route_idx] = route_dist(routes[max_route_idx])
                unassigned.add(cust)
            # reinsert using regret construction
            while unassigned:
                best_regret = -1e9
                best_cust = -1
                best_r_idx = -1
                best_pos = -1
                best_new_dist = -1.0
                best_cost_for_cust = None

                for cust in unassigned:
                    costs = []
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        for pos in range(1, len(route)):
                            new_max, new_dist = new_max_if_insert(cust, r_idx, pos)
                            costs.append((new_max, r_idx, pos, new_dist))
                    costs.sort(key=lambda x: x[0])
                    if len(costs) == 1:
                        regret = costs[0][0] * 2
                        best_candidate = costs[0]
                    else:
                        best = costs[0]
                        second_best = costs[1]
                        regret = second_best[0] - best[0]
                        best_candidate = best
                    tie_breaker = best_candidate[0]
                    if regret > best_regret or (regret == best_regret and (best_cost_for_cust is None or tie_breaker > best_cost_for_cust)):
                        best_regret = regret
                        best_cust = cust
                        best_r_idx = best_candidate[1]
                        best_pos = best_candidate[2]
                        best_new_dist = best_candidate[3]
                        best_cost_for_cust = tie_breaker
                    elif regret == best_regret and tie_breaker == best_cost_for_cust:
                        if cust < best_cust:
                            best_cust = cust
                            best_r_idx = best_candidate[1]
                            best_pos = best_candidate[2]
                            best_new_dist = best_candidate[3]

                routes[best_r_idx].insert(best_pos, best_cust)
                route_dists[best_r_idx] = best_new_dist
                unassigned.remove(best_cust)
            improved = True  # to continue local search after shake
        if not improved:
            break

    # Ensure exactly truck_count routes, each starting and ending at 0
    result = []
    for r in best_routes:
        if len(r) >= 2 and r[0] == 0 and r[-1] == 0:
            result.append(r)
        else:
            new_r = [0] + [c for c in r if c != 0] + [0]
            result.append(new_r)
    while len(result) < truck_count:
        result.append([0, 0])
    return result
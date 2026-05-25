import numpy as np
from copy import deepcopy

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    routes = [[depot, depot] for _ in range(truck_count)]
    unassigned = set(range(1, n))

    def route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    def best_insertion(route, customer):
        best_pos = -1
        best_delta = float('inf')
        for pos in range(1, len(route)):
            prev = route[pos-1]
            nxt = route[pos]
            delta = distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
        return best_pos, best_delta

    # Construction with 2-regret (fixed regret weight = 1)
    while unassigned:
        cust_info = []
        for cust in unassigned:
            deltas = []
            for ridx, route in enumerate(routes):
                pos, delta = best_insertion(route, cust)
                deltas.append(delta)
            sorted_deltas = sorted(enumerate(deltas), key=lambda x: x[1])
            best_ridx, best_delta = sorted_deltas[0]
            second_best_delta = sorted_deltas[1][1] if len(sorted_deltas) > 1 else best_delta
            regret = second_best_delta - best_delta
            score = regret + best_delta  # standard regret-2
            cust_info.append((score, regret, best_delta, cust, best_ridx))
        # Primary sort by score (lower better), tie-break by higher regret
        cust_info.sort(key=lambda x: (x[0], -x[1], x[2], x[3]))
        _, _, _, cust, ridx = cust_info[0]
        pos, _ = best_insertion(routes[ridx], cust)
        routes[ridx].insert(pos, cust)
        unassigned.remove(cust)

    route_dists = [route_distance(r) for r in routes]
    best_routes = [r[:] for r in routes]
    best_max = max(route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    max_iter = 10 * n
    for _ in range(max_iter):
        current_max = max(route_dists)
        max_idx = route_dists.index(current_max)
        improved = False

        # Inter-route: remove a customer from longest route and reinsert elsewhere
        route_max = routes[max_idx]
        best_removal = None
        best_new_max = current_max
        best_new_routes = None
        for i in range(1, len(route_max)-1):
            cust = route_max[i]
            new_route = route_max[:i] + route_max[i+1:]
            new_dist = route_distance(new_route)
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_other_dist = route_distance(new_other)
                    candidate_max = max(new_dist, new_other_dist)
                    if candidate_max < best_new_max:
                        best_new_max = candidate_max
                        best_removal = (max_idx, i, other_idx, pos)
                        best_new_routes = (new_route, new_other)
        if best_new_max < current_max:
            max_idx_rem, i, other_idx, pos = best_removal
            routes[max_idx_rem] = best_new_routes[0]
            routes[other_idx] = best_new_routes[1]
            route_dists[max_idx_rem] = route_distance(routes[max_idx_rem])
            route_dists[other_idx] = route_distance(routes[other_idx])
            improved = True

        if not improved:
            # Intra-route 2-opt on longest route
            route = routes[max_idx]
            best_route = route[:]
            best_dist = route_dists[max_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist:
                        best_route = new_route
                        best_dist = new_dist
            if best_dist < route_dists[max_idx]:
                routes[max_idx] = best_route
                route_dists[max_idx] = best_dist
                improved = True

        if improved:
            new_max = max(route_dists)
            if new_max < current_max:
                best_routes = [r[:] for r in routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
        else:
            break

    while len(best_routes) < truck_count:
        best_routes.append([depot, depot])
    return best_routes
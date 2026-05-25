import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(route, cust):
        best_pos = -1
        best_delta = float('inf')
        for pos in range(1, len(route)):
            prev = route[pos-1]
            nxt = route[pos]
            delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
        return best_pos, best_delta

    def construct_routes_deterministic():
        routes = [[depot, depot] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            best_regret = -float('inf')
            best_cust = None
            best_ridx = None
            best_pos = None
            best_delta = None
            for cust in unassigned:
                deltas = []
                positions = []
                for ridx, route in enumerate(routes):
                    pos, delta = best_insertion(route, cust)
                    deltas.append(delta)
                    positions.append(pos)
                sorted_deltas = sorted(enumerate(deltas), key=lambda x: x[1])
                top3 = sorted_deltas[:3]
                if len(top3) < 3:
                    regret = sum(d for _, d in top3) - top3[0][1]
                else:
                    regret = top3[2][1] - top3[0][1]
                # tie-breaking: if regret equal, prefer smaller delta, then smaller cust
                if regret > best_regret or (regret == best_regret and (best_delta is None or deltas[top3[0][0]] < best_delta)):
                    best_regret = regret
                    best_cust = cust
                    best_ridx = top3[0][0]
                    best_pos = positions[best_ridx]
                    best_delta = deltas[best_ridx]
            routes[best_ridx].insert(best_pos, best_cust)
            unassigned.remove(best_cust)
        return routes

    def steepest_descent(routes):
        route_dists = [route_distance(r) for r in routes]
        improved = True
        # limit iterations to avoid infinite loops
        max_iter_ls = n * truck_count
        iter_ls = 0
        while improved and iter_ls < max_iter_ls:
            improved = False
            iter_ls += 1
            # first try to reduce the current max route
            max_dist = max(route_dists)
            # consider all routes, but prioritize max routes
            for ridx in sorted(range(truck_count), key=lambda x: -route_dists[x]):
                route = routes[ridx]
                # Inter-route relocate: try moving each customer to another route
                for i in range(1, len(route)-1):
                    cust = route[i]
                    new_route = route[:i] + route[i+1:]
                    new_dist = route_distance(new_route)
                    for other_idx in range(truck_count):
                        if other_idx == ridx:
                            continue
                        other_route = routes[other_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_dist = route_distance(new_other)
                            candidate_max = max(new_dist, new_other_dist)
                            if candidate_max < max_dist:
                                # tie-breaking: smallest max, then smallest cust
                                if candidate_max < max_dist or (candidate_max == max_dist and cust < route[i]):
                                    routes[ridx] = new_route
                                    routes[other_idx] = new_other
                                    route_dists[ridx] = new_dist
                                    route_dists[other_idx] = new_other_dist
                                    max_dist = candidate_max
                                    improved = True
                # Inter-route swap: try swapping customers
                for i in range(1, len(route)-1):
                    cust1 = route[i]
                    for other_idx in range(truck_count):
                        if other_idx == ridx:
                            continue
                        other_route = routes[other_idx]
                        for j in range(1, len(other_route)-1):
                            cust2 = other_route[j]
                            new_route = route[:i] + [cust2] + route[i+1:]
                            new_other = other_route[:j] + [cust1] + other_route[j+1:]
                            new_dist = route_distance(new_route)
                            new_other_dist = route_distance(new_other)
                            candidate_max = max(new_dist, new_other_dist)
                            if candidate_max < max_dist:
                                routes[ridx] = new_route
                                routes[other_idx] = new_other
                                route_dists[ridx] = new_dist
                                route_dists[other_idx] = new_other_dist
                                max_dist = candidate_max
                                improved = True
                # Intra-route 2-opt
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < route_dists[ridx]:
                            route_dists[ridx] = new_dist
                            routes[ridx] = new_route
                            max_dist = max(route_dists)
                            improved = True
        return routes, route_dists

    # Initial solution
    routes = construct_routes_deterministic()
    routes, route_dists = steepest_descent(routes)
    best_routes = [r[:] for r in routes]
    best_max = max(route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # ILS loops
    max_iter = min(5, n)
    for _ in range(max_iter - 1):
        # Perturb best solution: relocate one customer from longest route to another route
        perturb_routes = [r[:] for r in best_routes]
        perturb_dists = [route_distance(r) for r in perturb_routes]
        # find longest route(s)
        max_dist = max(perturb_dists)
        longest_indices = [i for i, d in enumerate(perturb_dists) if d == max_dist]
        if longest_indices:
            ridx = random.choice(longest_indices)
            route = perturb_routes[ridx]
            if len(route) > 2:
                # choose a random customer from that route
                i = random.randint(1, len(route)-2)
                cust = route[i]
                # remove customer from its route
                new_route = route[:i] + route[i+1:]
                new_dist = route_distance(new_route)
                # insert into a random other route
                other_idx = random.choice([idx for idx in range(truck_count) if idx != ridx])
                other_route = perturb_routes[other_idx]
                pos = random.randint(1, len(other_route)-1)
                new_other = other_route[:pos] + [cust] + other_route[pos:]
                new_other_dist = route_distance(new_other)
                perturb_routes[ridx] = new_route
                perturb_routes[other_idx] = new_other
                perturb_dists[ridx] = new_dist
                perturb_dists[other_idx] = new_other_dist
        # Local search on perturbed solution
        routes, route_dists = steepest_descent(perturb_routes)
        current_max = max(route_dists)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass

    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([depot, depot])
    return best_routes
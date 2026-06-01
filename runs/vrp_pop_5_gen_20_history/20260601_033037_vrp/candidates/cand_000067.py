import numpy as np
import heapq
import itertools
import collections

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Special case: truck_count >= number of customers
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return best_routes

    # Farthest-first seed selection
    seeds = []
    seed0 = max(customers, key=lambda x: distance_matrix[0, x])
    seeds.append(seed0)
    while len(seeds) < truck_count:
        best_cust = None
        best_min_dist = -1.0
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c, s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and (best_cust is None or c < best_cust)):
                best_min_dist = min_dist
                best_cust = c
        if best_cust is not None:
            seeds.append(best_cust)
        else:
            break

    # Regret-2 assignment: assign customers to nearest seed in order of decreasing regret
    unassigned = [c for c in customers if c not in seeds]
    clusters = [[] for _ in range(truck_count)]
    while unassigned:
        # Compute regret for each unassigned customer
        best = []
        for c in unassigned:
            dists = [distance_matrix[c, s] for s in seeds]
            sorted_dists = sorted(dists)
            regret = sorted_dists[1] - sorted_dists[0] if len(sorted_dists) > 1 else 0.0
            nearest_idx = dists.index(sorted_dists[0])
            best.append((regret, c, nearest_idx))
        # Sort by regret descending, then customer index ascending
        best.sort(key=lambda x: (-x[0], x[1]))
        regret, c, seed_idx = best[0]
        clusters[seed_idx].append(c)
        unassigned.remove(c)
    for i, s in enumerate(seeds):
        clusters[i].append(s)

    # Cheapest insertion to build routes from clusters
    def build_routes_from_clusters(clusters):
        routes = []
        for cl in clusters:
            if not cl:
                routes.append([0, 0])
            else:
                route = [0, 0]
                remaining = list(cl)
                while remaining:
                    best_cust = None
                    best_pos = None
                    best_cost = float('inf')
                    for c in remaining:
                        for pos in range(1, len(route)):
                            delta = (distance_matrix[route[pos-1], c] +
                                     distance_matrix[c, route[pos]] -
                                     distance_matrix[route[pos-1], route[pos]])
                            if delta < best_cost or (delta == best_cost and (best_cust is None or c < best_cust)):
                                best_cost = delta
                                best_cust = c
                                best_pos = pos
                    route.insert(best_pos, best_cust)
                    remaining.remove(best_cust)
                routes.append(route)
        return routes

    routes = build_routes_from_clusters(clusters)
    report_best_vrp(routes)

    # Local search loop
    max_iter = min(200, n * truck_count)
    for _ in range(max_iter):
        improved = False
        # Identify longest route
        dists = [route_distance(r) for r in routes]
        longest_idx = max(range(truck_count), key=lambda i: (dists[i], i))

        # Inter-route relocate from longest route
        route_l = routes[longest_idx]
        interior = route_l[1:-1]
        if interior:
            for cust in interior:
                for other_idx in range(truck_count):
                    if other_idx == longest_idx:
                        continue
                    other_route = routes[other_idx]
                    best_pos = None
                    best_delta = float('inf')
                    for pos in range(1, len(other_route)):
                        prev = other_route[pos-1]
                        nxt = other_route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if delta < best_delta - 1e-12 or (abs(delta - best_delta) < 1e-12 and (best_pos is None or pos < best_pos)):
                            best_delta = delta
                            best_pos = pos
                    new_routes = [list(r) for r in routes]
                    new_routes[longest_idx].remove(cust)
                    new_routes[other_idx].insert(best_pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_max - 1e-12:
                        routes = new_routes
                        report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue

        # Inter-route swap between longest and another route
        if interior:
            for other_idx in range(truck_count):
                if other_idx == longest_idx:
                    continue
                other_interior = routes[other_idx][1:-1]
                if not other_interior:
                    continue
                for cust_l in interior:
                    for cust_o in other_interior:
                        new_routes = [list(r) for r in routes]
                        idx_l = new_routes[longest_idx].index(cust_l)
                        idx_o = new_routes[other_idx].index(cust_o)
                        new_routes[longest_idx][idx_l] = cust_o
                        new_routes[other_idx][idx_o] = cust_l
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max - 1e-12:
                            routes = new_routes
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue

        # Intra-route Or-opt
        for idx in range(truck_count):
            route = routes[idx]
            if len(route) <= 3:
                continue
            best_route = list(route)
            best_dist = route_distance(route)
            # segment length 1, 2, 3
            for seg_len in range(1, min(4, len(route)-1)):
                for start in range(1, len(route)-seg_len):
                    end = start + seg_len - 1
                    segment = route[start:end+1]
                    remaining = route[:start] + route[end+1:]
                    for pos in range(1, len(remaining)):
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        if new_route[0] != 0 or new_route[-1] != 0:
                            continue
                        d = route_distance(new_route)
                        if d < best_dist - 1e-12:
                            best_dist = d
                            best_route = new_route
            if best_dist < route_distance(route) - 1e-12:
                routes[idx] = best_route
                new_max = max(route_distance(r) for r in routes)
                if new_max < best_max - 1e-12:
                    report_best_vrp(routes)
                improved = True
                break
        if improved:
            continue

        # Ruin-recreate: remove up to 10% of customers from longest route and reinsert
        if len(interior) > 2:
            remove_count = max(1, len(interior) // 10)
            # Remove the first 'remove_count' customers from interior (by index order)
            to_remove = interior[:remove_count]
            new_routes = [list(r) for r in routes]
            removed_customers = []
            for cust in to_remove:
                new_routes[longest_idx].remove(cust)
                removed_customers.append(cust)
            # Reinsert all removed customers using cheapest insertion across all routes
            for cust in removed_customers:
                best_route_idx = None
                best_pos = None
                best_cost = float('inf')
                for r_idx, r in enumerate(new_routes):
                    for pos in range(1, len(r)):
                        delta = (distance_matrix[r[pos-1], cust] +
                                 distance_matrix[cust, r[pos]] -
                                 distance_matrix[r[pos-1], r[pos]])
                        if delta < best_cost - 1e-12 or (abs(delta - best_cost) < 1e-12 and (best_route_idx is None or r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                            best_cost = delta
                            best_route_idx = r_idx
                            best_pos = pos
                new_routes[best_route_idx].insert(best_pos, cust)
            new_max = max(route_distance(r) for r in new_routes)
            if new_max < best_max - 1e-12:
                routes = new_routes
                report_best_vrp(routes)
                improved = True
        if not improved:
            break

    final_routes = best_routes if best_routes is not None else routes
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes
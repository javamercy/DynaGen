import numpy as np
import math
import heapq
import itertools
import collections

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    
    # Helper functions
    def route_dist(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_max_dist(routes, route_dists):
        return max(route_dists)

    def get_customer_positions(route):
        # return indices of customers (excluding depots)
        return list(range(1, len(route)-1))

    # Initial solution construction with regret-based insertion
    def construct_solution():
        routes = [[depot, depot] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            best_regret = -1e9
            best_cust = -1
            best_r_idx = -1
            best_pos = -1
            best_new_dist = -1
            best_tie_cost = None
            for cust in unassigned:
                costs = []
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_dist = route_dists[r_idx] + delta
                        cand_max = new_dist
                        for i in range(truck_count):
                            if i != r_idx:
                                cand_max = max(cand_max, route_dists[i])
                        costs.append((cand_max, r_idx, pos, new_dist))
                costs.sort(key=lambda x: (x[0], x[1], x[2]))
                best = costs[0]
                if len(costs) == 1:
                    regret = best[0] * 2
                else:
                    second = costs[1]
                    regret = second[0] - best[0]
                if regret > best_regret or (regret == best_regret and (best_tie_cost is None or best[0] > best_tie_cost)):
                    best_regret = regret
                    best_cust = cust
                    best_r_idx = best[1]
                    best_pos = best[2]
                    best_new_dist = best[3]
                    best_tie_cost = best[0]
                elif regret == best_regret and best[0] == best_tie_cost and cust < best_cust:
                    best_cust = cust
                    best_r_idx = best[1]
                    best_pos = best[2]
                    best_new_dist = best[3]
            routes[best_r_idx].insert(best_pos, best_cust)
            route_dists[best_r_idx] = best_new_dist
            unassigned.remove(best_cust)
        return routes, route_dists

    # Local search blocks
    def try_relocate(routes, route_dists, best_max, best_routes):
        improved = False
        for r_from in range(truck_count):
            if len(routes[r_from]) <= 2:
                continue
            for pos_from in get_customer_positions(routes[r_from]):
                cust = routes[r_from][pos_from]
                new_route_from = routes[r_from][:pos_from] + routes[r_from][pos_from+1:]
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
                            return improved, best_max, best_routes
        return improved, best_max, best_routes

    def try_swap(routes, route_dists, best_max, best_routes):
        improved = False
        for r1 in range(truck_count):
            if len(routes[r1]) <= 2:
                continue
            for pos1 in get_customer_positions(routes[r1]):
                cust1 = routes[r1][pos1]
                for r2 in range(r1+1, truck_count):
                    if len(routes[r2]) <= 2:
                        continue
                    for pos2 in get_customer_positions(routes[r2]):
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
                            return improved, best_max, best_routes
        return improved, best_max, best_routes

    def try_intra_2opt(routes, route_dists, best_max, best_routes):
        improved = False
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 4:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_dists[r_idx]:
                        cand_max = max(new_dist, max(route_dists[:r_idx] + route_dists[r_idx+1:]))
                        if cand_max < best_max:
                            best_max = cand_max
                            best_routes = [list(r) for r in routes]
                            best_routes[r_idx] = new_route
                            routes[r_idx] = new_route
                            route_dists[r_idx] = new_dist
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            return improved, best_max, best_routes
        return improved, best_max, best_routes

    def try_cross_2opt(routes, route_dists, best_max, best_routes):
        improved = False
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for r2 in range(r1+1, truck_count):
                route2 = routes[r2]
                if len(route2) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        # Try 2-opt*: replace edges (i,i+1) and (j,j+1) with (i,j+1) and (j,i+1)
                        new_route1 = route1[:i+1] + route2[j+1:]
                        new_route2 = route2[:j+1] + route1[i+1:]
                        # Ensure routes start and end at depot
                        # new_route1 and new_route2 are valid because they preserve depot
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
                            return improved, best_max, best_routes
        return improved, best_max, best_routes

    def local_search(routes, route_dists, best_max, best_routes):
        max_iter = 2 * (n - 1)
        for _ in range(max_iter):
            improved = False
            improved, best_max, best_routes = try_relocate(routes, route_dists, best_max, best_routes)
            if improved: continue
            improved, best_max, best_routes = try_swap(routes, route_dists, best_max, best_routes)
            if improved: continue
            improved, best_max, best_routes = try_cross_2opt(routes, route_dists, best_max, best_routes)
            if improved: continue
            improved, best_max, best_routes = try_intra_2opt(routes, route_dists, best_max, best_routes)
            if not improved:
                break
        return best_routes, best_max, route_dists

    # Build initial solution
    routes, route_dists = construct_solution()
    best_routes = [list(r) for r in routes]
    best_max = compute_max_dist(routes, route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # First round of local search
    best_routes, best_max, route_dists = local_search(routes, route_dists, best_max, best_routes)

    # Restart mechanism (shake and re-optimize)
    max_restarts = 3
    for restart in range(max_restarts):
        # Deterministic shake: remove a fraction of customers based on restart count
        num_remove = max(1, (n - 1) // 10)
        # Remove customers with smallest indices
        all_custs = list(range(1, n))
        to_remove = set(sorted(all_custs)[:num_remove])
        # Build partial solution by removing those customers
        new_routes = [list(r) for r in best_routes]
        new_route_dists = list(route_dists)
        removed = []
        for r_idx in range(truck_count):
            route = new_routes[r_idx]
            new_route = [0]
            for c in route[1:-1]:
                if c in to_remove:
                    removed.append(c)
                else:
                    new_route.append(c)
            new_route.append(0)
            new_routes[r_idx] = new_route
            new_route_dists[r_idx] = route_dist(new_route)
        # Reinsert removed customers using regret insertion
        unassigned = set(removed)
        while unassigned:
            best_regret = -1e9
            best_cust = -1
            best_r_idx = -1
            best_pos = -1
            best_new_dist = -1
            best_tie_cost = None
            for cust in unassigned:
                costs = []
                for r_idx in range(truck_count):
                    route = new_routes[r_idx]
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_dist = new_route_dists[r_idx] + delta
                        cand_max = new_dist
                        for i in range(truck_count):
                            if i != r_idx:
                                cand_max = max(cand_max, new_route_dists[i])
                        costs.append((cand_max, r_idx, pos, new_dist))
                costs.sort(key=lambda x: (x[0], x[1], x[2]))
                best = costs[0]
                if len(costs) == 1:
                    regret = best[0] * 2
                else:
                    second = costs[1]
                    regret = second[0] - best[0]
                if regret > best_regret or (regret == best_regret and (best_tie_cost is None or best[0] > best_tie_cost)):
                    best_regret = regret
                    best_cust = cust
                    best_r_idx = best[1]
                    best_pos = best[2]
                    best_new_dist = best[3]
                    best_tie_cost = best[0]
                elif regret == best_regret and best[0] == best_tie_cost and cust < best_cust:
                    best_cust = cust
                    best_r_idx = best[1]
                    best_pos = best[2]
                    best_new_dist = best[3]
            new_routes[best_r_idx].insert(best_pos, best_cust)
            new_route_dists[best_r_idx] = best_new_dist
            unassigned.remove(best_cust)
        new_max = compute_max_dist(new_routes, new_route_dists)
        # Run local search on new solution
        new_best_routes, new_best_max, new_route_dists = local_search(new_routes, new_route_dists, new_max, [list(r) for r in new_routes])
        if new_best_max < best_max:
            best_max = new_best_max
            best_routes = [list(r) for r in new_best_routes]
            route_dists = list(new_route_dists)
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass

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
import numpy as np
import math
import heapq
import itertools
import collections

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    
    def route_dist(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_max_dist(route_dists):
        return max(route_dists)

    def get_customer_positions(route):
        return list(range(1, len(route)-1))

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

    def get_all_relocate_moves(routes, route_dists):
        moves = []  # each element: (new_max, r_from, pos_from, r_to, pos_to, new_routes, new_dists)
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
                        new_routes = [list(r) for r in routes]
                        new_routes[r_from] = new_route_from
                        new_routes[r_to] = new_route_to
                        new_dists = list(route_dists)
                        new_dists[r_from] = new_dist_from
                        new_dists[r_to] = new_dist_to
                        moves.append((cand_max, r_from, pos_from, r_to, pos_to, new_routes, new_dists))
        return moves

    def get_all_swap_moves(routes, route_dists):
        moves = []
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
                        new_routes = [list(r) for r in routes]
                        new_routes[r1] = new_route1
                        new_routes[r2] = new_route2
                        new_dists = list(route_dists)
                        new_dists[r1] = new_dist1
                        new_dists[r2] = new_dist2
                        moves.append((cand_max, r1, pos1, r2, pos2, new_routes, new_dists))
        return moves

    def get_all_cross2opt_moves(routes, route_dists):
        moves = []
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
                        new_route1 = route1[:i+1] + route2[j+1:]
                        new_route2 = route2[:j+1] + route1[i+1:]
                        new_dist1 = route_dist(new_route1)
                        new_dist2 = route_dist(new_route2)
                        cand_max = max(new_dist1, new_dist2)
                        for r in range(truck_count):
                            if r != r1 and r != r2:
                                cand_max = max(cand_max, route_dists[r])
                        new_routes = [list(r) for r in routes]
                        new_routes[r1] = new_route1
                        new_routes[r2] = new_route2
                        new_dists = list(route_dists)
                        new_dists[r1] = new_dist1
                        new_dists[r2] = new_dist2
                        moves.append((cand_max, r1, i, r2, j, new_routes, new_dists))
        return moves

    def get_all_intra2opt_moves(routes, route_dists):
        moves = []
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 4:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_dists[r_idx]:  # only if improves individual route distance
                        cand_max = max(new_dist, max(route_dists[:r_idx] + route_dists[r_idx+1:]))
                        new_routes = [list(r) for r in routes]
                        new_routes[r_idx] = new_route
                        new_dists = list(route_dists)
                        new_dists[r_idx] = new_dist
                        moves.append((cand_max, r_idx, i, j, new_routes, new_dists))
        return moves

    def apply_best_move(moves, routes, route_dists):
        if not moves:
            return False, routes, route_dists, False
        # sort by new_max, then deterministic tie-break (r_from, pos_from, etc.)
        # we'll just take min by new_max; tie-breaking not crucial
        best_move = min(moves, key=lambda x: (x[0], x[1:]))
        new_routes = best_move[5]
        new_dists = best_move[6]
        # update in place
        for i in range(truck_count):
            routes[i] = new_routes[i]
            route_dists[i] = new_dists[i]
        return True, routes, route_dists, True

    def local_search(routes, route_dists, best_max, best_routes):
        max_iter = 2 * (n - 1)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            # Relocate moves
            moves = get_all_relocate_moves(routes, route_dists)
            applied, routes, route_dists, imp = apply_best_move(moves, routes, route_dists)
            if applied:
                new_max = compute_max_dist(route_dists)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                improved = True
                iter_count += 1
                continue
            # Swap moves
            moves = get_all_swap_moves(routes, route_dists)
            applied, routes, route_dists, imp = apply_best_move(moves, routes, route_dists)
            if applied:
                new_max = compute_max_dist(route_dists)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                improved = True
                iter_count += 1
                continue
            # Cross-2opt moves
            moves = get_all_cross2opt_moves(routes, route_dists)
            applied, routes, route_dists, imp = apply_best_move(moves, routes, route_dists)
            if applied:
                new_max = compute_max_dist(route_dists)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                improved = True
                iter_count += 1
                continue
            # Intra-2opt moves
            moves = get_all_intra2opt_moves(routes, route_dists)
            applied, routes, route_dists, imp = apply_best_move(moves, routes, route_dists)
            if applied:
                new_max = compute_max_dist(route_dists)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                improved = True
                iter_count += 1
                continue
        return best_routes, best_max, route_dists

    # Build initial solution
    routes, route_dists = construct_solution()
    best_routes = [list(r) for r in routes]
    best_max = compute_max_dist(route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # First round of local search
    best_routes, best_max, route_dists = local_search(routes, route_dists, best_max, best_routes)

    # Restart mechanism (shake and re-optimize)
    max_restarts = 3
    for restart in range(max_restarts):
        # Find longest route (by distance)
        max_dist = max(route_dists)
        longest_indices = [i for i, d in enumerate(route_dists) if d == max_dist]
        # Pick the longest route with most customers (deterministic)
        longest_idx = max(longest_indices, key=lambda i: len(best_routes[i]))
        route_long = best_routes[longest_idx]
        if len(route_long) <= 3:
            continue
        # Remove a fraction of customers from longest route
        num_remove = max(1, (n-1) // 10)
        # Remove customers with smallest index in that route
        customers_in_route = [c for c in route_long if c != 0]
        customers_in_route.sort()
        to_remove = set(customers_in_route[:num_remove])
        # Build new routes by removing those customers
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
        new_max = compute_max_dist(new_route_dists)
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
import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    unassigned = set(customers)

    # Initialize routes
    routes = [[depot, depot] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count

    def route_dist(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def update_route_dists():
        for i, r in enumerate(routes):
            route_dists[i] = route_dist(r)

    update_route_dists()

    def new_max_if_insert(cust, r_idx, pos):
        route = routes[r_idx]
        prev = route[pos-1]
        next_ = route[pos]
        delta = distance_matrix[prev, cust] + distance_matrix[cust, next_] - distance_matrix[prev, next_]
        new_dist = route_dists[r_idx] + delta
        candidate_max = new_dist
        for i in range(truck_count):
            if i == r_idx:
                continue
            if route_dists[i] > candidate_max:
                candidate_max = route_dists[i]
        return candidate_max, new_dist

    # Construction
    while unassigned:
        best_regret = -1e9
        best_cust = -1
        best_r_idx = -1
        best_pos = -1
        best_new_dist = -1
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
            elif regret == best_regret and tie_breaker == best_cost_for_cust and cust < best_cust:
                best_cust = cust
                best_r_idx = best_candidate[1]
                best_pos = best_candidate[2]
                best_new_dist = best_candidate[3]

        routes[best_r_idx].insert(best_pos, best_cust)
        route_dists[best_r_idx] = best_new_dist
        unassigned.remove(best_cust)

    # Post-construction record best
    current_max = max(route_dists)
    best_routes = [list(r) for r in routes]
    best_max = current_max
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # Local search with restart
    max_iter_no_improve = 5  # restart after this many non-improving iterations
    no_improve = 0
    while no_improve < max_iter_no_improve:
        improved = False

        # Inter-route relocate
        for r_from in range(truck_count):
            if len(routes[r_from]) <= 2:
                continue
            cust_positions = list(range(1, len(routes[r_from])-1))
            for pos_from in cust_positions:
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
                        candidate_max = max(new_dist_from, new_dist_to)
                        for r in range(truck_count):
                            if r != r_from and r != r_to:
                                candidate_max = max(candidate_max, route_dists[r])
                        if candidate_max < best_max:
                            best_max = candidate_max
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
            no_improve = 0
            continue

        # Inter-route swap
        for r1 in range(truck_count):
            if len(routes[r1]) <= 2:
                continue
            for pos1 in range(1, len(routes[r1])-1):
                cust1 = routes[r1][pos1]
                for r2 in range(r1+1, truck_count):
                    if len(routes[r2]) <= 2:
                        continue
                    for pos2 in range(1, len(routes[r2])-1):
                        cust2 = routes[r2][pos2]
                        new_route1 = list(routes[r1])
                        new_route2 = list(routes[r2])
                        new_route1[pos1] = cust2
                        new_route2[pos2] = cust1
                        new_dist1 = route_dist(new_route1)
                        new_dist2 = route_dist(new_route2)
                        candidate_max = max(new_dist1, new_dist2)
                        for r in range(truck_count):
                            if r != r1 and r != r2:
                                candidate_max = max(candidate_max, route_dists[r])
                        if candidate_max < best_max:
                            best_max = candidate_max
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
            no_improve = 0
            continue

        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 4:
                continue
            best_imp = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_dists[r_idx]:
                        candidate_max = max(new_dist, max(route_dists[:r_idx] + route_dists[r_idx+1:]))
                        if candidate_max < best_max:
                            best_max = candidate_max
                            best_routes = [list(r) for r in routes]
                            best_routes[r_idx] = new_route
                            routes[r_idx] = new_route
                            route_dists[r_idx] = new_dist
                            improved = True
                            best_imp = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                if best_imp:
                    break
            if improved:
                break

        if improved:
            no_improve = 0
            continue

        # Cross-route 2-opt* (also known as inter-route 2-opt: exchange tails)
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for r2 in range(r1+1, truck_count):
                route2 = routes[r2]
                if len(route2) <= 2:
                    continue
                for i in range(1, len(route1)-1):  # i is cut in route1 (after depot)
                    for j in range(1, len(route2)-1):  # j is cut in route2
                        new_route1 = route1[:i] + route2[j:]
                        new_route2 = route2[:j] + route1[i:]
                        # Ensure routes start and end at depot; need to fix if last node not depot?
                        # But routes currently have depot at ends, so new_route1 ends with the tail of route2 which ends in depot, similarly for new_route2.
                        new_dist1 = route_dist(new_route1)
                        new_dist2 = route_dist(new_route2)
                        candidate_max = max(new_dist1, new_dist2)
                        for r in range(truck_count):
                            if r != r1 and r != r2:
                                candidate_max = max(candidate_max, route_dists[r])
                        if candidate_max < best_max:
                            best_max = candidate_max
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
            no_improve = 0
            continue

        # If not improved, restart by re-randomizing some routes
        no_improve += 1
        if no_improve >= max_iter_no_improve:
            break
        # Restart: break longest route and redistribute its customers
        # Find route with maximum distance
        longest_idx = np.argmax(route_dists)
        longest_route = routes[longest_idx]
        # Remove longest route's customers (except depots)
        custs_longest = longest_route[1:-1]
        # Reinsert them using regret heuristic with a twist: randomize order
        random.shuffle(custs_longest)
        for cust in custs_longest:
            # Remove cust from its current route (already removed)
            # Insert into any route using regret-based with slight randomness (choose random among top 2 best insertions)
            costs = []
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_max, new_dist = new_max_if_insert(cust, r_idx, pos)
                    costs.append((new_max, r_idx, pos, new_dist))
            costs.sort(key=lambda x: x[0])
            # Choose random among first min(len(costs), 2) options
            choice = random.randint(0, min(1, len(costs)-1))
            best_candidate = costs[choice]
            r_idx = best_candidate[1]
            pos = best_candidate[2]
            routes[r_idx].insert(pos, cust)
            route_dists[r_idx] = best_candidate[3]  # new_dist (we stored new_dist, but we need new_dist from new_max_if_insert, actually new_max_if_insert returns (new_max, new_dist), we stored (new_max, r_idx, pos, new_dist) in costs, so best_candidate[3] is new_dist)
        update_route_dists()
        # Record best if better
        current_max = max(route_dists)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass

    # Ensure exactly truck_count routes
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
import numpy as np
import math
from collections import deque

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_max_min(routes, dists):
        max_dist = max(dists)
        min_dist = min(dists)
        return max_dist, min_dist

    # Regret-2 construction with squared penalty
    routes = [[0, 0] for _ in range(truck_count)]
    dists = [0.0 for _ in range(truck_count)]
    customers = list(range(1, n))

    # Precompute costs for each customer to each route
    for cust in customers:
        best_cost = float('inf')
        best_pos = -1
        best_route = -1
        for r in range(truck_count):
            route = routes[r]
            best_pos_r = -1
            best_cost_r = float('inf')
            for i in range(1, len(route)):
                cost = (distance_matrix[route[i-1], cust] + distance_matrix[cust, route[i]] - distance_matrix[route[i-1], route[i]]) ** 2
                if cost < best_cost_r:
                    best_cost_r = cost
                    best_pos_r = i
            if best_cost_r < best_cost:
                best_cost = best_cost_r
                best_pos = best_pos_r
                best_route = r
        routes[best_route].insert(best_pos, cust)
        dists[best_route] = route_distance(routes[best_route])

    # Post-construction imbalance reduction
    improved = True
    while improved:
        max_dist, min_dist = compute_max_min(routes, dists)
        if max_dist - min_dist < 1e-9:
            break
        max_idx = dists.index(max_dist)
        min_idx = dists.index(min_dist)
        route_max = routes[max_idx]
        best_improve = 0
        best_cust = None
        best_pos = -1
        for idx in range(1, len(route_max)-1):
            cust = route_max[idx]
            # compute cost of removing cust
            new_route_max = route_max[:idx] + route_max[idx+1:]
            new_dist_max = route_distance(new_route_max)
            # try inserting into min route
            route_min = routes[min_idx]
            best_cost = float('inf')
            best_pos_min = -1
            for i in range(1, len(route_min)):
                cost = distance_matrix[route_min[i-1], cust] + distance_matrix[cust, route_min[i]] - distance_matrix[route_min[i-1], route_min[i]]
                if cost < best_cost:
                    best_cost = cost
                    best_pos_min = i
            new_dist_min = dists[min_idx] + best_cost
            new_max = max(new_dist_max, new_dist_min)
            if new_max < max_dist - 1e-9:
                improvement = max_dist - new_max
                if improvement > best_improve:
                    best_improve = improvement
                    best_cust = cust
                    best_pos = best_pos_min
        if best_improve > 0:
            # remove from max, insert into min
            route_max.remove(best_cust)
            routes[min_idx].insert(best_pos, best_cust)
            dists[max_idx] = route_distance(route_max)
            dists[min_idx] = route_distance(routes[min_idx])
        else:
            improved = False

    best_routes = [list(r) for r in routes]
    best_max = max(dists)
    report_best_vrp(routes)

    # Tabu search setup
    n_cust = n - 1
    max_iters = 10 * n_cust * truck_count
    tabu_tenure = 5
    # tabu: dictionary mapping (customer, from_route) -> iteration when tabu expires
    tabu = {}
    iteration = 0
    no_improve = 0
    max_no_improve = 50
    restart_triggered = False

    while iteration < max_iters:
        improved = False
        iteration += 1

        # Neighborhood: relocate
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for idx in range(1, len(route1)-1):
                cust = route1[idx]
                if (cust, r1) in tabu and tabu[(cust, r1)] > iteration:
                    continue
                new_route1 = route1[:idx] + route1[idx+1:]
                new_dist1 = route_distance(new_route1)
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    route2 = routes[r2]
                    # find best insertion in r2
                    best_cost = float('inf')
                    best_pos = -1
                    for i in range(1, len(route2)):
                        cost = distance_matrix[route2[i-1], cust] + distance_matrix[cust, route2[i]] - distance_matrix[route2[i-1], route2[i]]
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = i
                    new_dist2 = dists[r2] + best_cost
                    other_dists = [dists[i] for i in range(truck_count) if i not in (r1, r2)]
                    new_max = max(new_dist1, new_dist2, *other_dists)
                    if new_max < best_max - 1e-9:
                        # accept
                        routes[r1] = new_route1
                        routes[r2] = route2[:best_pos] + [cust] + route2[best_pos:]
                        dists[r1] = new_dist1
                        dists[r2] = new_dist2
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(routes)
                        tabu[(cust, r2)] = iteration + tabu_tenure
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            no_improve = 0
            continue

        # Neighborhood: swap
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust1 = route1[idx1]
                if (cust1, r1) in tabu and tabu[(cust1, r1)] > iteration:
                    continue
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        cust2 = route2[idx2]
                        if (cust2, r2) in tabu and tabu[(cust2, r2)] > iteration:
                            continue
                        new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                        new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_dists = [dists[i] for i in range(truck_count) if i not in (r1, r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < best_max - 1e-9:
                            routes[r1] = new_route1
                            routes[r2] = new_route2
                            dists[r1] = new_dist1
                            dists[r2] = new_dist2
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(routes)
                            tabu[(cust1, r2)] = iteration + tabu_tenure
                            tabu[(cust2, r1)] = iteration + tabu_tenure
                            improved = True
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
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            best_improve = 0
            best_i = best_j = -1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < dists[r] - 1e-9:
                        improvement = dists[r] - new_dist
                        if improvement > best_improve:
                            best_improve = improvement
                            best_i, best_j = i, j
            if best_improve > 0:
                new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                routes[r] = new_route
                dists[r] = route_distance(new_route)
                new_max = max(dists)
                if new_max < best_max - 1e-9:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(routes)
                improved = True
                break
        if improved:
            no_improve = 0
            continue

        # Cross-route 2-opt
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
                        new1 = route1[:i+1] + route2[j+1:]
                        new2 = route2[:j+1] + route1[i+1:]
                        new_dist1 = route_distance(new1)
                        new_dist2 = route_distance(new2)
                        other_dists = [dists[k] for k in range(truck_count) if k not in (r1, r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < best_max - 1e-9:
                            routes[r1] = new1
                            routes[r2] = new2
                            dists[r1] = new_dist1
                            dists[r2] = new_dist2
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(routes)
                            improved = True
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

        # No improvement, check for restart
        no_improve += 1
        if no_improve >= max_no_improve and not restart_triggered:
            # Restart from best solution with perturbation
            restart_triggered = True
            routes = [list(r) for r in best_routes]
            dists = [route_distance(r) for r in routes]
            best_max = max(dists)
            # Perturb: for each customer, try to move to a different route (deterministic order)
            for cust in sorted(range(1, n)):  # deterministic order
                # find current route of cust
                cur_route = None
                for r in range(truck_count):
                    if cust in routes[r]:
                        cur_route = r
                        break
                if cur_route is None:
                    continue
                # evaluate move to other routes
                best_new_max = float('inf')
                best_target = -1
                best_pos = -1
                for r2 in range(truck_count):
                    if r2 == cur_route:
                        continue
                    route2 = routes[r2]
                    # find best insertion
                    for i in range(1, len(route2)):
                        cost = distance_matrix[route2[i-1], cust] + distance_matrix[cust, route2[i]] - distance_matrix[route2[i-1], route2[i]]
                        new_dist2 = dists[r2] + cost
                        # compute new dist for cur_route after removal
                        idx = routes[cur_route].index(cust)
                        new_route_cur = routes[cur_route][:idx] + routes[cur_route][idx+1:]
                        new_dist_cur = route_distance(new_route_cur)
                        other_dists = [dists[k] for k in range(truck_count) if k not in (cur_route, r2)]
                        new_max = max(new_dist_cur, new_dist2, *other_dists)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_target = r2
                            best_pos = i
                if best_target != -1 and best_new_max < max(dists) - 1e-9:
                    # perform move
                    idx = routes[cur_route].index(cust)
                    routes[cur_route] = routes[cur_route][:idx] + routes[cur_route][idx+1:]
                    routes[best_target].insert(best_pos, cust)
                    dists[cur_route] = route_distance(routes[cur_route])
                    dists[best_target] = route_distance(routes[best_target])
            # update best if improved
            new_max = max(dists)
            if new_max < best_max - 1e-9:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(routes)
            no_improve = 0
            # reset tabu
            tabu.clear()
            restart_triggered = False

    return best_routes
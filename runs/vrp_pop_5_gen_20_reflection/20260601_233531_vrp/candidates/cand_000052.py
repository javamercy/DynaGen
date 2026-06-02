import math
import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def max_route(routes):
        return max(route_distance(r) for r in routes)

    # Regret-2 construction
    def construct():
        unvisited = set(range(1, n))
        routes = [[0, 0] for _ in range(truck_count)]
        while unvisited:
            best_cust = None
            best_regret = -1
            best_route = None
            best_pos = None
            for cust in unvisited:
                costs = []
                for idx, route in enumerate(routes):
                    best_cost = float('inf')
                    best_pos_local = -1
                    for pos in range(1, len(route)):
                        cost = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        if cost < best_cost:
                            best_cost = cost
                            best_pos_local = pos
                    costs.append((best_cost, idx, best_pos_local))
                if len(costs) >= 2:
                    costs.sort(key=lambda x: x[0])
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = costs[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_cost, best_route, best_pos = min(costs, key=lambda x: x[0])
            # Insert best_cust into best_route at best_pos
            routes[best_route].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        # Ensure all routes start and end at 0 (already)
        return routes

    # Local search with 2-opt*, relocate, swap
    def local_search(routes):
        improved = True
        while improved:
            improved = False
            # Intra-route 2-opt
            for idx in range(len(routes)):
                route = routes[idx]
                if len(route) <= 3:
                    continue
                best_delta = 0
                best_i = best_k = None
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        if k - i == 1:
                            continue
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[k], route[k+1]]
                        new = distance_matrix[route[i-1], route[k]] + distance_matrix[route[i], route[k+1]]
                        delta = new - old
                        if delta < best_delta - 1e-12:
                            best_delta = delta
                            best_i, best_k = i, k
                if best_delta < -1e-12:
                    i, k = best_i, best_k
                    routes[idx][i:k+1] = reversed(routes[idx][i:k+1])
                    improved = True
                    report_best_vrp(routes)
            # Inter-route relocate
            for i_route in range(len(routes)):
                route_i = routes[i_route]
                for pos in range(1, len(route_i)-1):
                    cust = route_i.pop(pos)
                    best_cost = float('inf')
                    best_j = -1
                    best_k = -1
                    for j_route in range(len(routes)):
                        if j_route == i_route:
                            continue
                        route_j = routes[j_route]
                        for k in range(1, len(route_j)):
                            cost = distance_matrix[route_j[k-1], cust] + distance_matrix[cust, route_j[k]] - distance_matrix[route_j[k-1], route_j[k]]
                            if cost < best_cost:
                                best_cost = cost
                                best_j = j_route
                                best_k = k
                    # Revert removal
                    route_i.insert(pos, cust)
                    if best_j != -1:
                        # Compute new distances
                        old_total = total_distance(routes)
                        old_max = max_route(routes)
                        # Perform move temporarily
                        route_i.pop(pos)
                        routes[best_j].insert(best_k, cust)
                        new_total = total_distance(routes)
                        new_max = max_route(routes)
                        if new_max < old_max - 1e-12 or (abs(new_max - old_max) < 1e-12 and new_total < old_total - 1e-12):
                            improved = True
                            report_best_vrp(routes)
                        else:
                            # revert
                            routes[best_j].pop(best_k)
                            route_i.insert(pos, cust)
            # Inter-route swap
            for i_route in range(len(routes)):
                route_i = routes[i_route]
                for pos_i in range(1, len(route_i)-1):
                    for j_route in range(i_route+1, len(routes)):
                        route_j = routes[j_route]
                        for pos_j in range(1, len(route_j)-1):
                            # Swap customers
                            cust_i = route_i[pos_i]
                            cust_j = route_j[pos_j]
                            # Compute delta
                            old_total = total_distance(routes)
                            old_max = max_route(routes)
                            route_i[pos_i] = cust_j
                            route_j[pos_j] = cust_i
                            new_total = total_distance(routes)
                            new_max = max_route(routes)
                            if new_max < old_max - 1e-12 or (abs(new_max - old_max) < 1e-12 and new_total < old_total - 1e-12):
                                improved = True
                                report_best_vrp(routes)
                            else:
                                route_i[pos_i] = cust_i
                                route_j[pos_j] = cust_j
            # Inter-route 2-opt*
            for i_route in range(len(routes)):
                route_i = routes[i_route]
                for j_route in range(i_route+1, len(routes)):
                    route_j = routes[j_route]
                    if len(route_i) <= 2 or len(route_j) <= 2:
                        continue
                    for i in range(1, len(route_i)-1):
                        for j in range(1, len(route_j)-1):
                            # Swap tails: route_i after i and route_j after j
                            # New route_i: route_i[:i+1] + route_j[j+1:]
                            # New route_j: route_j[:j+1] + route_i[i+1:]
                            # Check feasibility (each starts and ends at 0? ends will be 0 because last node is 0? Actually route ends with 0, but swapping tails will keep 0 at end? route_i ends with 0, route_j ends with 0. After swap, new route_i ends with last of route_j which is 0, and new route_j ends with last of route_i which is 0. So still ends with 0. Starts with 0 as well. So fine.
                            old_i = route_i[:]
                            old_j = route_j[:]
                            new_route_i = route_i[:i+1] + route_j[j+1:]
                            new_route_j = route_j[:j+1] + route_i[i+1:]
                            # Check if new routes are valid (each has at least 0 at both ends)
                            if len(new_route_i) < 2 or len(new_route_j) < 2:
                                continue
                            # Compute new total and max
                            old_total = total_distance(routes)
                            old_max = max_route(routes)
                            routes[i_route] = new_route_i
                            routes[j_route] = new_route_j
                            new_total = total_distance(routes)
                            new_max = max_route(routes)
                            if new_max < old_max - 1e-12 or (abs(new_max - old_max) < 1e-12 and new_total < old_total - 1e-12):
                                improved = True
                                report_best_vrp(routes)
                                break
                            else:
                                routes[i_route] = old_i
                                routes[j_route] = old_j
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Perturbation: weighted ruin and regret-2 recreate
    def perturb(routes, removal_ratio):
        n_customers = sum(len(r)-2 for r in routes)
        n_remove = max(1, int(removal_ratio * n_customers))
        # Weighted removal: prefer customers with high distance to depot
        all_cust = []
        for idx, r in enumerate(routes):
            for cust in r[1:-1]:
                weight = distance_matrix[0, cust]
                all_cust.append((weight, idx, cust))
        all_cust.sort(reverse=True)
        removed = []
        for _, idx, cust in all_cust[:n_remove]:
            r = routes[idx]
            pos = r.index(cust)
            r.pop(pos)
            removed.append(cust)
        # Regret-2 reinsertion
        while removed:
            best_cust = None
            best_regret = -1
            best_route = None
            best_pos = None
            for cust in removed:
                costs = []
                for idx, r in enumerate(routes):
                    best_cost = float('inf')
                    best_pos_local = -1
                    for pos in range(1, len(r)):
                        cost = distance_matrix[r[pos-1], cust] + distance_matrix[cust, r[pos]] - distance_matrix[r[pos-1], r[pos]]
                        if cost < best_cost:
                            best_cost = cost
                            best_pos_local = pos
                    costs.append((best_cost, idx, best_pos_local))
                if len(costs) >= 2:
                    costs.sort(key=lambda x: x[0])
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = costs[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_cost, best_route, best_pos = min(costs, key=lambda x: x[0])
            routes[best_route].insert(best_pos, best_cust)
            removed.remove(best_cust)
        return routes

    # Main
    best_routes = None
    best_max = float('inf')
    best_total = float('inf')
    for trial in range(1):  # single construction
        routes = construct()
        total = total_distance(routes)
        maxd = max_route(routes)
        report_best_vrp(routes)
        routes = local_search(routes)
        # Perturbation cycles with decreasing removal ratio
        ratios = [0.5, 0.4, 0.3, 0.2, 0.1]
        for ratio in ratios:
            routes = perturb(routes, ratio)
            routes = local_search(routes)
        # Final check
        if max_route(routes) < best_max - 1e-12 or (abs(max_route(routes)-best_max) < 1e-12 and total_distance(routes) < best_total):
            best_max = max_route(routes)
            best_total = total_distance(routes)
            best_routes = [r[:] for r in routes]
    return best_routes
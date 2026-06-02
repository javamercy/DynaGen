import numpy as np
import random
import math
from copy import deepcopy

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    random.seed(42)  # deterministic
    
    # ---------- helper functions ----------
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def deep_copy_routes(routes):
        return [list(r) for r in routes]

    # ---------- construction: regret-2 ----------
    def regret2_insertion(customers, routes, route_dists):
        unassigned = set(customers)
        while unassigned:
            best_cust = None
            best_pos = None
            best_route = None
            best_regret = -float('inf')
            best_cost = float('inf')
            for cust in unassigned:
                insertion_costs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        old_d = route_dists[r_idx]
                        removed = distance_matrix[route[pos-1], route[pos]]
                        added = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]]
                        new_d = old_d - removed + added
                        insertion_costs.append((new_d, r_idx, pos, new_d - old_d))
                # sort by cost delta (increase in route distance)
                insertion_costs.sort(key=lambda x: x[1] if len(insertion_costs) > 1 else 0)  # placeholder, actually sort by delta
                # properly
                insertion_costs.sort(key=lambda x: x[3])
                if len(insertion_costs) == 0:
                    continue
                best_cost_cust = insertion_costs[0][0]
                if len(insertion_costs) >= 2:
                    regret = insertion_costs[1][3] - insertion_costs[0][3]
                else:
                    regret = 0
                if regret > best_regret or (regret == best_regret and best_cost_cust < best_cost):
                    best_regret = regret
                    best_cost = best_cost_cust
                    best_cust = cust
                    best_route = insertion_costs[0][1]
                    best_pos = insertion_costs[0][2]
            # insert best customer
            routes[best_route].insert(best_pos, best_cust)
            route_dists[best_route] = best_cost
            unassigned.remove(best_cust)
        return routes

    # initial empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    customers = list(range(1, n))
    # randomize order for deterministic seed
    random.shuffle(customers)
    routes = regret2_insertion(customers, routes, route_dists)
    best_routes = deep_copy_routes(routes)
    best_max = max_distance(routes)
    current_routes = deep_copy_routes(routes)
    current_max = best_max

    # ---------- SA parameters ----------
    T0 = 1000.0
    T_end = 1e-6
    cooling = 0.999
    T = T0
    iteration = 0
    max_iter = 200 * n

    # ---------- ruin operators ----------
    def ruin_random(routes, num_remove):
        removed = []
        for _ in range(num_remove):
            # pick a non-empty route weighted by route distance?
            non_empty = [i for i, r in enumerate(routes) if len(r) > 2]
            if not non_empty:
                break
            r_idx = random.choice(non_empty)
            route = routes[r_idx]
            pos = random.randint(1, len(route)-2)
            cust = route.pop(pos)
            removed.append(cust)
        return removed

    def ruin_worst(routes, num_remove):
        # remove customers that cause largest detour
        detours = []
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                next_ = route[pos+1]
                detour = distance_matrix[prev, cust] + distance_matrix[cust, next_] - distance_matrix[prev, next_]
                detours.append((detour, r_idx, pos, cust))
        detours.sort(key=lambda x: x[0], reverse=True)
        removed = []
        for i in range(min(num_remove, len(detours))):
            _, r_idx, pos, cust = detours[i]
            if routes[r_idx][pos] == cust:  # still there?
                routes[r_idx].pop(pos)
                removed.append(cust)
        return removed

    def ruin_cluster(routes, num_remove):
        # remove a continuous segment from one route
        non_empty = [i for i, r in enumerate(routes) if len(r) > 2]
        if not non_empty:
            return []
        r_idx = random.choice(non_empty)
        route = routes[r_idx]
        if len(route) <= 2:
            return []
        start = random.randint(1, len(route)-2)
        end = min(start + num_remove, len(route)-1)
        removed = []
        for pos in range(end-1, start-1, -1):
            if pos < len(route)-1 and pos > 0:
                removed.append(route.pop(pos))
        return removed

    # ---------- recreate: regret-2 ----------
    def recreate(removed, routes, route_dists):
        # reinsert removed customers using regret-2
        unassigned = set(removed)
        while unassigned:
            best_cust = None
            best_pos = None
            best_route = None
            best_regret = -float('inf')
            best_cost_delta = float('inf')
            for cust in unassigned:
                deltas = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        old_d = route_dists[r_idx]
                        removed_d = distance_matrix[route[pos-1], route[pos]]
                        added_d = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]]
                        new_d = old_d - removed_d + added_d
                        delta = new_d - old_d
                        deltas.append((delta, r_idx, pos))
                if not deltas:
                    continue
                deltas.sort(key=lambda x: x[0])
                best_delta = deltas[0][0]
                if len(deltas) >= 2:
                    regret = deltas[1][0] - deltas[0][0]
                else:
                    regret = 0
                if regret > best_regret or (regret == best_regret and best_delta < best_cost_delta):
                    best_regret = regret
                    best_cost_delta = best_delta
                    best_cust = cust
                    best_route = deltas[0][1]
                    best_pos = deltas[0][2]
            # insert
            routes[best_route].insert(best_pos, best_cust)
            route_dists[best_route] += best_cost_delta
            unassigned.remove(best_cust)
        return routes

    # ---------- balancing local search ----------
    def balance(routes):
        # try to reduce max distance by swapping customers between routes
        improved = True
        while improved:
            improved = False
            for i in range(truck_count):
                for j in range(i+1, truck_count):
                    for pos_i in range(1, len(routes[i])-1):
                        for pos_j in range(1, len(routes[j])-1):
                            new_route_i = routes[i].copy()
                            new_route_j = routes[j].copy()
                            # swap
                            new_route_i[pos_i], new_route_j[pos_j] = new_route_j[pos_j], new_route_i[pos_i]
                            # check feasibility? no need, all nodes distinct
                            d_i = route_distance(new_route_i)
                            d_j = route_distance(new_route_j)
                            new_max = max(d_i, d_j)
                            old_max = max(route_distance(routes[i]), route_distance(routes[j]))
                            if new_max < old_max:
                                routes[i] = new_route_i
                                routes[j] = new_route_j
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break
        return routes

    # ---------- main loop ----------
    while iteration < max_iter and T > T_end:
        iteration += 1
        old_routes = deep_copy_routes(current_routes)
        old_max = current_max
        # ruin
        num_remove = random.randint(int(0.1*n), int(0.3*n))
        op = random.choices(['random', 'worst', 'cluster'], weights=[0.4,0.4,0.2])[0]
        if op == 'random':
            removed = ruin_random(old_routes, num_remove)
        elif op == 'worst':
            removed = ruin_worst(old_routes, num_remove)
        else:
            removed = ruin_cluster(old_routes, num_remove)
        # remove from current routes (already removed in ruin)
        # update route_dists for current routes
        current_route_dists = [route_distance(r) for r in old_routes]
        # recreate
        new_routes = deep_copy_routes(old_routes)
        new_route_dists = list(current_route_dists)
        if removed:
            new_routes = recreate(removed, new_routes, new_route_dists)
        # optionally balance
        if random.random() < 0.2:
            new_routes = balance(new_routes)
        new_max = max(new_route_dists)
        # acceptance
        if new_max < current_max or random.random() < math.exp((current_max - new_max) / T):
            current_routes = deep_copy_routes(new_routes)
            current_max = new_max
            if new_max < best_max:
                best_routes = deep_copy_routes(new_routes)
                best_max = new_max
                report_best_vrp(best_routes)
        T *= cooling
        # update current_route_dists for next iteration
        # already updated in recreate

    report_best_vrp(best_routes)
    return best_routes
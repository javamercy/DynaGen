import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    # Initial construction: greedy insertion minimizing max distance
    customers = list(range(1, n))
    random.shuffle(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route)
                candidate_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                candidate_total = sum(route_dists[:t]) + new_dist + sum(route_dists[t+1:])
                if candidate_max < best_max or (candidate_max == best_max and candidate_total < best_total):
                    best_max = candidate_max
                    best_total = candidate_total
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)
        route_dists[best_truck] = route_distance(routes[best_truck])

    current_routes = [list(r) for r in routes]
    current_dists = list(route_dists)
    current_max = max(current_dists)
    current_total = sum(current_dists)
    best_routes = [list(r) for r in routes]
    best_dists = list(route_dists)
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    max_iter = min(2000, 15 * n)  # reduced iterations but more exploitation
    removal_fraction = 0.3
    num_removals = max(1, int(removal_fraction * (n - 1)))
    T0 = best_max * 0.1  # lower initial temperature
    T = T0

    def worst_removal(routes, dists, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if len(route) <= 2:
                continue
            base = dists[t]
            for i in range(1, len(route) - 1):
                prev = route[i-1]
                nxt = route[i+1]
                with_ = dist[prev, route[i]] + dist[route[i], nxt]
                without = dist[prev, nxt]
                contrib = with_ - without
                all_contribs.append((-contrib, t, i, route[i]))
        all_contribs.sort(key=lambda x: x[0])
        to_remove = set()
        for _, t, i, cust in all_contribs[:num_removals]:
            to_remove.add(cust)
        new_routes = []
        new_dists = []
        for t, route in enumerate(routes):
            new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
            new_routes.append(new_route)
            new_dists.append(route_distance(new_route))
        return list(to_remove), new_routes, new_dists

    def regret3_repair(routes, dists, unassigned):
        routes = [list(r) for r in routes]
        dists = list(dists)
        unassigned = list(unassigned)
        current_max_local = max(dists)
        while unassigned:
            best_info = None
            for cust in unassigned:
                costs = []
                for t, route in enumerate(routes):
                    old_dist = dists[t]
                    best_route_cost = float('inf')
                    best_pos = None
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_distance(new_route)
                        delta = new_dist - old_dist
                        if delta < best_route_cost:
                            best_route_cost = delta
                            best_pos = pos
                    costs.append((best_route_cost, t, best_pos))
                costs.sort(key=lambda x: x[0])
                regret = sum(c[0] for c in costs[1:]) - costs[0][0] if len(costs) >= 2 else float('inf')
                # Actually compute as (sum of top 3) - 3*best? For regret-3, we need three best
                # If num trucks < 3, adjust
                k = min(3, len(costs))
                regret = sum(c[0] for c in costs[1:k]) - (k-1)*costs[0][0]
                best_truck = costs[0][1]
                best_pos = costs[0][2]
                best_delta = costs[0][0]
                # Evaluate new max and total
                old_dists_list = list(dists)
                new_dists_list = list(dists)
                new_dists_list[best_truck] += best_delta
                new_max = max(new_dists_list)
                new_total = sum(new_dists_list)
                if best_info is None or regret > best_info[0] or (regret == best_info[0] and (new_max < best_info[1] or (new_max == best_info[1] and new_total < best_info[2]))):
                    best_info = (regret, new_max, new_total, cust, best_truck, best_pos, best_delta)
            regret, new_max, new_total, cust, best_truck, best_pos, best_delta = best_info
            routes[best_truck].insert(best_pos, cust)
            dists[best_truck] += best_delta
            if dists[best_truck] > current_max_local:
                current_max_local = dists[best_truck]
            unassigned.remove(cust)
        return routes, dists

    for it in range(max_iter):
        # Always use worst removal and regret-3 repair for exploitation
        removed, partial, partial_dists = worst_removal(current_routes, current_dists, num_removals)
        new_routes, new_dists = regret3_repair(partial, partial_dists, removed)

        new_max = max(new_dists)
        new_total = sum(new_dists)
        delta = new_max - current_max
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_routes = [list(r) for r in new_routes]
            current_dists = list(new_dists)
            current_max = new_max
            current_total = new_total
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes]
                best_dists = list(new_dists)
                report_best_vrp(best_routes)

        T = T0 * math.exp(-it / (max_iter / 2.0))

    # Post-optimization: intra-route 2-opt and inter-route relocate/swap
    # Run at most 3 passes of best-improvement local search
    for _ in range(3):
        improved = False
        # Intra-route 2-opt
        for t, route in enumerate(best_routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dists[t] - 1e-9:
                        new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                        new_total = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                            best_routes[t] = new_route
                            best_dists[t] = new_dist
                            best_max = new_max
                            best_total = new_total
                            report_best_vrp(best_routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Inter-route relocate: move a customer from one route to another
        for t1 in range(truck_count):
            if len(best_routes[t1]) <= 2:
                continue
            for i in range(1, len(best_routes[t1])-1):
                cust = best_routes[t1][i]
                for t2 in range(truck_count):
                    if t1 == t2:
                        continue
                    route2 = best_routes[t2]
                    for pos in range(1, len(route2)):
                        new_route1 = best_routes[t1][:i] + best_routes[t1][i+1:]
                        new_route2 = route2[:pos] + [cust] + route2[pos:]
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        new_dists_list = list(best_dists)
                        new_dists_list[t1] = new_dist1
                        new_dists_list[t2] = new_dist2
                        new_max = max(new_dists_list)
                        new_total = sum(new_dists_list)
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                            best_routes[t1] = new_route1
                            best_routes[t2] = new_route2
                            best_dists[t1] = new_dist1
                            best_dists[t2] = new_dist2
                            best_max = new_max
                            best_total = new_total
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Inter-route swap: exchange two customers
        for t1 in range(truck_count):
            if len(best_routes[t1]) <= 2:
                continue
            for i in range(1, len(best_routes[t1])-1):
                cust1 = best_routes[t1][i]
                for t2 in range(truck_count):
                    if t1 == t2:
                        continue
                    route2 = best_routes[t2]
                    for j in range(1, len(route2)-1):
                        cust2 = route2[j]
                        new_route1 = list(best_routes[t1])
                        new_route1[i] = cust2
                        new_route2 = list(route2)
                        new_route2[j] = cust1
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        new_dists_list = list(best_dists)
                        new_dists_list[t1] = new_dist1
                        new_dists_list[t2] = new_dist2
                        new_max = max(new_dists_list)
                        new_total = sum(new_dists_list)
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                            best_routes[t1] = new_route1
                            best_routes[t2] = new_route2
                            best_dists[t1] = new_dist1
                            best_dists[t2] = new_dist2
                            best_max = new_max
                            best_total = new_total
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_routes
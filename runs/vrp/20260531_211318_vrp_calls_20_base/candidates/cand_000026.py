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
    
    # ---------- Initial construction (greedy min max) ----------
    customers = list(range(1, n))
    routes = [[] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = -1
        best_pos = -1
        best_max = float('inf')
        best_total = float('inf')
        for t in range(truck_count):
            route = routes[t]
            for pos in range(len(route)+1):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance([0] + new_route + [0])
                other_max = 0.0
                for tt in range(truck_count):
                    if tt != t:
                        other_max = max(other_max, route_dists[tt])
                new_max = max(new_dist, other_max)
                new_total = new_dist + sum(route_dists[tt] for tt in range(truck_count) if tt != t)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        route = routes[best_truck]
        route.insert(best_pos, cust)
        route_dists[best_truck] = route_distance([0] + route + [0])
    
    current_routes = [[0] + r + [0] for r in routes]
    best_routes = [list(r) for r in current_routes]
    best_max = max(route_distance(r) for r in best_routes)
    best_total = sum(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    # ---------- Helper functions ----------
    def worst_removal(routes, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if len(route) <= 2:
                continue
            base = route_distance(route)
            for i in range(1, len(route)-1):
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
        for t, route in enumerate(routes):
            new_routes.append([0] + [c for c in route[1:-1] if c not in to_remove] + [0])
        return list(to_remove), new_routes
    
    def random_removal(routes, num_removals):
        all_customers = [c for r in routes for c in r[1:-1]]
        random.shuffle(all_customers)
        to_remove = set(all_customers[:num_removals])
        new_routes = []
        for route in routes:
            new_routes.append([0] + [c for c in route[1:-1] if c not in to_remove] + [0])
        return list(to_remove), new_routes
    
    def greedy_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        for cust in unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_truck = -1
            best_pos = -1
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_routes = routes[:t] + [new_route] + routes[t+1:]
                    new_max_val = max(route_distance(r) for r in new_routes)
                    new_total_val = sum(route_distance(r) for r in new_routes)
                    if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                        best_max = new_max_val
                        best_total = new_total_val
                        best_truck = t
                        best_pos = pos
            routes[best_truck].insert(best_pos, cust)
        return routes
    
    def regret2_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        while unassigned:
            best_info = None
            for cust in unassigned:
                best_max = float('inf')
                best_total = float('inf')
                second_best_max = float('inf')
                second_best_total = float('inf')
                best_truck = -1
                best_pos = -1
                for t, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        new_r = route[:pos] + [cust] + route[pos:]
                        new_routes = routes[:t] + [new_r] + routes[t+1:]
                        new_max_val = max(route_distance(r) for r in new_routes)
                        new_total_val = sum(route_distance(r) for r in new_routes)
                        if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                            second_best_max = best_max
                            second_best_total = best_total
                            best_max = new_max_val
                            best_total = new_total_val
                            best_truck = t
                            best_pos = pos
                        elif new_max_val < second_best_max or (new_max_val == second_best_max and new_total_val < second_best_total):
                            second_best_max = new_max_val
                            second_best_total = new_total_val
                if second_best_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_best_max - best_max
                if best_info is None:
                    best_info = (regret, best_max, best_total, cust, best_truck, best_pos)
                else:
                    if regret > best_info[0] or (regret == best_info[0] and (best_max > best_info[1] or (best_max == best_info[1] and best_total > best_info[2]))):
                        best_info = (regret, best_max, best_total, cust, best_truck, best_pos)
            regret, best_max_val, best_total_val, cust, best_truck, best_pos = best_info
            routes[best_truck].insert(best_pos, cust)
            unassigned.remove(cust)
        return routes
    
    # ---------- ALNS phase ----------
    max_iter = 50 * n
    removal_fraction = 0.3
    num_removals = max(1, int(removal_fraction * (n-1)))
    T0 = best_max / 2.0
    T = T0
    current_routes = [list(r) for r in best_routes]
    current_max = best_max
    current_total = best_total
    
    for it in range(max_iter):
        destroy_op = random.choice([0, 1])
        repair_op = random.choice([0, 1])
        if destroy_op == 0:
            removed, partial = worst_removal(current_routes, num_removals)
        else:
            removed, partial = random_removal(current_routes, num_removals)
        if repair_op == 0:
            new_routes = greedy_repair(partial, removed)
        else:
            new_routes = regret2_repair(partial, removed)
        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)
        delta = new_max - current_max
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_routes = [list(r) for r in new_routes]
            current_max = new_max
            current_total = new_total
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)
        T = T0 * (1 - it / max_iter)
    
    # ---------- Deterministic local search (backbone moves) on best ----------
    routes = [list(r[1:-1]) for r in best_routes]  # remove depot for easier manipulation
    route_dists = [route_distance(r) for r in best_routes]
    max_dist = best_max
    max_passes = 10 * n * truck_count
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        def max_other(exclude):
            d = 0.0
            for t, rd in enumerate(route_dists):
                if t not in exclude:
                    d = max(d, rd)
            return d
        # Balancing: move from longest to shortest
        max_idx = max(range(truck_count), key=lambda i: route_dists[i])
        min_idx = min(range(truck_count), key=lambda i: route_dists[i])
        if max_idx != min_idx and route_dists[max_idx] > 0:
            route_max = routes[max_idx]
            route_min = routes[min_idx]
            for cust in route_max:
                new_max_route = [c for c in route_max if c != cust]
                new_max_dist = route_distance([0] + new_max_route + [0])
                best_pos = 0
                best_inc = float('inf')
                for pos in range(len(route_min)+1):
                    new_route_min = route_min[:pos] + [cust] + route_min[pos:]
                    new_min_dist = route_distance([0] + new_route_min + [0])
                    inc = new_min_dist - route_dists[min_idx]
                    if inc < best_inc:
                        best_inc = inc
                        best_pos = pos
                new_min_route = route_min[:best_pos] + [cust] + route_min[best_pos:]
                new_min_dist = route_distance([0] + new_min_route + [0])
                other_max = max_other({max_idx, min_idx})
                new_max = max(new_max_dist, new_min_dist, other_max)
                if new_max < max_dist - 1e-9:
                    routes[max_idx] = new_max_route
                    routes[min_idx] = new_min_route
                    route_dists[max_idx] = new_max_dist
                    route_dists[min_idx] = new_min_dist
                    max_dist = new_max
                    improved = True
                    full = [[0] + r + [0] for r in routes]
                    report_best_vrp(full)
                    break
        if improved:
            continue
        # Relocate
        for t_from in range(truck_count):
            if not routes[t_from]:
                continue
            for i in range(len(routes[t_from])):
                cust = routes[t_from][i]
                old_dist_from = route_dists[t_from]
                new_dist_from = route_distance([0] + [c for idx,c in enumerate(routes[t_from]) if idx != i] + [0])
                for t_to in range(truck_count):
                    if t_to == t_from:
                        continue
                    route_to = routes[t_to]
                    for j in range(len(route_to)+1):
                        new_route_to = route_to[:j] + [cust] + route_to[j:]
                        new_dist_to = route_distance([0] + new_route_to + [0])
                        other_max = max_other({t_from, t_to})
                        new_max = max(new_dist_from, new_dist_to, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t_from].pop(i)
                            route_dists[t_from] = new_dist_from
                            routes[t_to].insert(j, cust)
                            route_dists[t_to] = new_dist_to
                            max_dist = new_max
                            improved = True
                            full = [[0] + r + [0] for r in routes]
                            report_best_vrp(full)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Swap
        for t1 in range(truck_count):
            if not routes[t1]:
                continue
            for i in range(len(routes[t1])):
                cust1 = routes[t1][i]
                for t2 in range(t1+1, truck_count):
                    if not routes[t2]:
                        continue
                    for j in range(len(routes[t2])):
                        cust2 = routes[t2][j]
                        new_route1 = routes[t1].copy()
                        new_route2 = routes[t2].copy()
                        new_route1[i] = cust2
                        new_route2[j] = cust1
                        new_dist1 = route_distance([0] + new_route1 + [0])
                        new_dist2 = route_distance([0] + new_route2 + [0])
                        other_max = max_other({t1, t2})
                        new_max = max(new_dist1, new_dist2, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            route_dists[t1] = new_dist1
                            route_dists[t2] = new_dist2
                            max_dist = new_max
                            improved = True
                            full = [[0] + r + [0] for r in routes]
                            report_best_vrp(full)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt within route
        for t in range(truck_count):
            route = routes[t]
            if len(route) < 2:
                continue
            for i in range(len(route)-1):
                for j in range(i+1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance([0] + new_route + [0])
                    if new_dist < route_dists[t] - 1e-9:
                        other_max = max_other({t})
                        new_max = max(new_dist, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t] = new_route
                            route_dists[t] = new_dist
                            max_dist = new_max
                            improved = True
                            full = [[0] + r + [0] for r in routes]
                            report_best_vrp(full)
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Cross-route exchange
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                route1 = routes[t1]
                route2 = routes[t2]
                if not route1 or not route2:
                    continue
                for i in range(len(route1)+1):
                    for j in range(len(route2)+1):
                        new_route1 = route1[:i] + route2[j:]
                        new_route2 = route2[:j] + route1[i:]
                        new_dist1 = route_distance([0] + new_route1 + [0])
                        new_dist2 = route_distance([0] + new_route2 + [0])
                        other_max = max_other({t1, t2})
                        new_max = max(new_dist1, new_dist2, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            route_dists[t1] = new_dist1
                            route_dists[t2] = new_dist2
                            max_dist = new_max
                            improved = True
                            full = [[0] + r + [0] for r in routes]
                            report_best_vrp(full)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    final_routes = [[0] + r + [0] for r in routes]
    return final_routes
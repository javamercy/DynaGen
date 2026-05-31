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

    def insertion_cost(route, cust, pos, curr_dist):
        if pos == 0:
            return curr_dist - dist[0, route[0]] + dist[0, cust] + dist[cust, route[0]]
        elif pos == len(route):
            return curr_dist - dist[route[-1], 0] + dist[route[-1], cust] + dist[cust, 0]
        else:
            prev = route[pos-1]
            nxt = route[pos]
            return curr_dist - dist[prev, nxt] + dist[prev, cust] + dist[cust, nxt]

    def removal_cost(route, i, curr_dist):
        if len(route) == 1:
            return 0.0
        if i == 0:
            return curr_dist - dist[0, route[0]] - dist[route[0], route[1]] + dist[0, route[1]]
        elif i == len(route)-1:
            return curr_dist - dist[route[-2], route[-1]] - dist[route[-1], 0] + dist[route[-2], 0]
        else:
            prev = route[i-1]
            nxt = route[i+1]
            return curr_dist - dist[prev, route[i]] - dist[route[i], nxt] + dist[prev, nxt]

    # Initial construction: greedy insertion minimizing max distance
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]  # store as list with depot already included?
    # We'll keep routes as list of customer lists (without depots) for easier manipulation, then convert at the end
    routes = [[] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = None
        best_pos = None
        best_new_max = float('inf')
        best_new_total = float('inf')
        for t in range(truck_count):
            route = routes[t]
            curr_dist = route_dists[t]
            for pos in range(len(route)+1):
                new_dist = insertion_cost(route, cust, pos, curr_dist)
                other_max = max(route_dists[:t] + route_dists[t+1:])
                new_max = max(new_dist, other_max)
                new_total = sum(route_dists[:t] + [new_dist] + route_dists[t+1:])
                if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                    best_new_max = new_max
                    best_new_total = new_total
                    best_truck = t
                    best_pos = pos
        route = routes[best_truck]
        route.insert(best_pos, cust)
        route_dists[best_truck] = insertion_cost(route, cust, best_pos, route_dists[best_truck] if best_pos != 0 else 0)  # recalc easier
        # Recompute route distance for that route
        routes[best_truck] = route
        route_dists[best_truck] = sum(dist[route[i], route[i+1]] for i in range(len(route)-1)) + dist[0, route[0]] + dist[route[-1], 0] if route else 0.0

    # Convert to full routes with depots
    def get_full_routes(routes):
        return [[0] + r + [0] for r in routes]

    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    current_routes = [list(r) for r in routes]
    current_max = best_max
    report_best_vrp(get_full_routes(best_routes))

    # ALNS parameters
    max_iter = 50 * n
    removal_fraction = 0.3
    num_removals = max(1, int(removal_fraction * (n-1)))
    T0 = best_max / 2.0
    T = T0
    # Operator weights
    destroy_weights = [1.0, 1.0, 1.0]  # worst, random, max-route
    repair_weights = [1.0, 1.0]  # greedy, regret2
    destroy_names = ['worst', 'random', 'maxroute']
    repair_names = ['greedy', 'regret2']
    success_destroy = [0.0, 0.0, 0.0]
    success_repair = [0.0, 0.0]
    total_destroy = [0.0, 0.0, 0.0]
    total_repair = [0.0, 0.0]
    alpha = 0.1  # response coefficient

    # Helper functions for ALNS
    def worst_removal(routes, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if not route:
                continue
            base = route_dists[t]
            for i, cust in enumerate(route):
                new_dist = removal_cost(route, i, base)
                contrib = base - new_dist  # reduction if removed
                all_contribs.append((contrib, t, i, cust))
        all_contribs.sort(key=lambda x: -x[0])
        to_remove = set()
        for _, t, i, cust in all_contribs[:num_removals]:
            to_remove.add(cust)
        new_routes = [[c for c in route if c not in to_remove] for route in routes]
        return list(to_remove), new_routes

    def random_removal(routes, num_removals):
        all_custs = [c for r in routes for c in r]
        random.shuffle(all_custs)
        to_remove = set(all_custs[:num_removals])
        new_routes = [[c for c in route if c not in to_remove] for route in routes]
        return list(to_remove), new_routes

    def max_route_removal(routes, num_removals):
        # remove from the longest route(s)
        dists = [route_dists[t] for t in range(truck_count)]
        sorted_idx = sorted(range(truck_count), key=lambda i: -dists[i])
        to_remove = set()
        remaining = num_removals
        for t in sorted_idx:
            route = routes[t]
            if not route:
                continue
            take = min(remaining, len(route))
            # take random customers from this route? Let's take the ones with highest removal contribution? Use worst inside this route
            if take > 0:
                contribs = []
                for i, cust in enumerate(route):
                    new_dist = removal_cost(route, i, dists[t])
                    contrib = dists[t] - new_dist
                    contribs.append((contrib, cust))
                contribs.sort(key=lambda x: -x[0])
                for _, cust in contribs[:take]:
                    to_remove.add(cust)
                remaining -= take
            if remaining <= 0:
                break
        new_routes = [[c for c in route if c not in to_remove] for route in routes]
        return list(to_remove), new_routes

    def greedy_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        # We'll compute route distances dynamically
        route_dists = [sum(dist[0, r[0]] + dist[r[-1], 0] + sum(dist[r[i], r[i+1]] for i in range(len(r)-1)) if r else 0.0) for r in routes]
        for cust in unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_truck = None
            best_pos = None
            for t in range(truck_count):
                route = routes[t]
                curr_dist = route_dists[t]
                for pos in range(len(route)+1):
                    new_dist = insertion_cost(route, cust, pos, curr_dist)
                    other_max = max(route_dists[:t] + route_dists[t+1:])
                    new_max = max(new_dist, other_max)
                    new_total = sum(route_dists[:t] + [new_dist] + route_dists[t+1:])
                    if new_max < best_max or (new_max == best_max and new_total < best_total):
                        best_max = new_max
                        best_total = new_total
                        best_truck = t
                        best_pos = pos
            routes[best_truck].insert(best_pos, cust)
            route_dists[best_truck] = insertion_cost(routes[best_truck], cust, best_pos, route_dists[best_truck] if best_pos != 0 else 0)  # need correct current dist before insertion? We'll recompute
            # Recompute route_dists[best_truck] properly
            r = routes[best_truck]
            route_dists[best_truck] = sum(dist[0, r[0]] + dist[r[-1], 0] + sum(dist[r[i], r[i+1]] for i in range(len(r)-1)) if r else 0.0)
        return routes, route_dists

    def regret2_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        route_dists = [sum(dist[0, r[0]] + dist[r[-1], 0] + sum(dist[r[i], r[i+1]] for i in range(len(r)-1)) if r else 0.0) for r in routes]
        while unassigned:
            best_info = None
            for cust in unassigned:
                best_max = float('inf')
                best_total = float('inf')
                second_best_max = float('inf')
                second_best_total = float('inf')
                best_truck = None
                best_pos = None
                for t in range(truck_count):
                    route = routes[t]
                    curr_dist = route_dists[t]
                    for pos in range(len(route)+1):
                        new_dist = insertion_cost(route, cust, pos, curr_dist)
                        other_max = max(route_dists[:t] + route_dists[t+1:])
                        new_max = max(new_dist, other_max)
                        new_total = sum(route_dists[:t] + [new_dist] + route_dists[t+1:])
                        if new_max < best_max or (new_max == best_max and new_total < best_total):
                            second_best_max = best_max
                            second_best_total = best_total
                            best_max = new_max
                            best_total = new_total
                            best_truck = t
                            best_pos = pos
                        elif new_max < second_best_max or (new_max == second_best_max and new_total < second_best_total):
                            second_best_max = new_max
                            second_best_total = new_total
                if second_best_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_best_max - best_max
                if best_info is None:
                    best_info = (regret, best_max, cust, best_truck, best_pos)
                else:
                    if regret > best_info[0] or (regret == best_info[0] and (best_max < best_info[1] or (best_max == best_info[1] and cust < best_info[2]))):
                        best_info = (regret, best_max, cust, best_truck, best_pos)
            regret, best_max_val, cust, best_truck, best_pos = best_info
            routes[best_truck].insert(best_pos, cust)
            r = routes[best_truck]
            route_dists[best_truck] = sum(dist[0, r[0]] + dist[r[-1], 0] + sum(dist[r[i], r[i+1]] for i in range(len(r)-1)) if r else 0.0)
            unassigned.remove(cust)
        return routes, route_dists

    # Main ALNS loop
    for it in range(max_iter):
        # Select operators proportional to weights
        destroy_op = random.choices(destroy_names, weights=destroy_weights, k=1)[0]
        repair_op = random.choices(repair_names, weights=repair_weights, k=1)[0]

        # Destroy
        if destroy_op == 'worst':
            removed, partial = worst_removal(current_routes, num_removals)
        elif destroy_op == 'random':
            removed, partial = random_removal(current_routes, num_removals)
        else:  # maxroute
            removed, partial = max_route_removal(current_routes, num_removals)

        # Repair
        if repair_op == 'greedy':
            new_routes, new_dists = greedy_repair(partial, removed)
        else:
            new_routes, new_dists = regret2_repair(partial, removed)

        # Evaluate
        new_max = max(new_dists)
        new_total = sum(new_dists)
        old_max = max(route_dists[t] for t in range(truck_count) if current_routes[t] else 0.0)  # careful: current_routes may have empty routes
        old_max = max([route_dists[t] for t in range(truck_count)] if any(current_routes) else 0.0)
        # Actually we need route_dists of current_routes
        current_dists = []
        for r in current_routes:
            if not r:
                current_dists.append(0.0)
            else:
                d = dist[0, r[0]] + dist[r[-1], 0]
                for i in range(len(r)-1):
                    d += dist[r[i], r[i+1]]
                current_dists.append(d)
        current_max = max(current_dists)
        current_total = sum(current_dists)

        delta = new_max - current_max
        accept = False
        if delta < 0 or (delta == 0 and new_total < current_total):
            accept = True
        elif random.random() < math.exp(-delta / max(T, 1e-9)):
            accept = True

        if accept:
            current_routes = [list(r) for r in new_routes]
            route_dists = new_dists.copy()
            # update best
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_dists)):
                best_max = new_max
                best_routes = [list(r) for r in current_routes]
                report_best_vrp(get_full_routes(best_routes))
            # Update operator success counts
            idx_d = destroy_names.index(destroy_op)
            idx_r = repair_names.index(repair_op)
            success_destroy[idx_d] += 1
            success_repair[idx_r] += 1

        total_destroy[destroy_names.index(destroy_op)] += 1
        total_repair[repair_names.index(repair_op)] += 1

        # Temperature update
        T = T0 * (1 - (it+1) / max_iter)

        # Periodically update weights
        if (it+1) % 100 == 0:
            for i in range(len(destroy_weights)):
                if total_destroy[i] > 0:
                    success_rate = success_destroy[i] / total_destroy[i]
                else:
                    success_rate = 0
                destroy_weights[i] = max(0.01, destroy_weights[i] * (1 - alpha) + alpha * (success_rate * len(destroy_names)))
                total_destroy[i] = 0
                success_destroy[i] = 0
            for i in range(len(repair_weights)):
                if total_repair[i] > 0:
                    success_rate = success_repair[i] / total_repair[i]
                else:
                    success_rate = 0
                repair_weights[i] = max(0.01, repair_weights[i] * (1 - alpha) + alpha * (success_rate * len(repair_names)))
                total_repair[i] = 0
                success_repair[i] = 0

    # Final local search for polishing (deterministic)
    # Use best_routes and route_dists from best state
    routes = [list(r) for r in best_routes]
    route_dists = []
    for r in routes:
        if not r:
            route_dists.append(0.0)
        else:
            d = dist[0, r[0]] + dist[r[-1], 0]
            for i in range(len(r)-1):
                d += dist[r[i], r[i+1]]
            route_dists.append(d)
    max_dist = max(route_dists)
    improved = True
    passes = 0
    max_passes = 5 * n * truck_count
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # Balancing: move from longest to shortest if reduces max
        max_idx = max(range(truck_count), key=lambda i: route_dists[i])
        min_idx = min(range(truck_count), key=lambda i: route_dists[i])
        if max_idx != min_idx and route_dists[max_idx] > 0:
            route_max = routes[max_idx]
            route_min = routes[min_idx]
            # pick cust with highest removal contribution in max route
            best_cust = None
            best_pos_in_min = None
            best_gain = -1e9
            for i, cust in enumerate(route_max):
                new_max_route = routes[max_idx][:i] + routes[max_idx][i+1:]
                new_max_dist = route_distance([0] + new_max_route + [0]) if new_max_route else 0.0
                old_min_dist = route_dists[min_idx]
                # find best insertion in min route
                for pos in range(len(route_min)+1):
                    new_min_route = route_min[:pos] + [cust] + route_min[pos:]
                    new_min_dist = route_distance([0] + new_min_route + [0]) if new_min_route else 0.0
                    other_max = max(route_dists[:max_idx] + route_dists[max_idx+1:min_idx] + route_dists[min_idx+1:], default=0.0)
                    new_max_val = max(new_max_dist, new_min_dist, other_max)
                    if new_max_val < max_dist - 1e-9:
                        # apply
                        routes[max_idx] = new_max_route
                        routes[min_idx] = new_min_route
                        route_dists[max_idx] = new_max_dist
                        route_dists[min_idx] = new_min_dist
                        max_dist = new_max_val
                        improved = True
                        break
                if improved:
                    break
        if improved:
            continue

        # Relocate
        for t_from in range(truck_count):
            if not routes[t_from]:
                continue
            for i, cust in enumerate(routes[t_from]):
                new_dist_from = removal_cost(routes[t_from], i, route_dists[t_from])
                for t_to in range(truck_count):
                    if t_to == t_from:
                        continue
                    for pos in range(len(routes[t_to])+1):
                        new_dist_to = insertion_cost(routes[t_to], cust, pos, route_dists[t_to])
                        other_max = max(route_dists[t] for t in range(truck_count) if t != t_from and t != t_to)
                        new_max_val = max(new_dist_from, new_dist_to, other_max)
                        if new_max_val < max_dist - 1e-9:
                            routes[t_from].pop(i)
                            route_dists[t_from] = new_dist_from
                            routes[t_to].insert(pos, cust)
                            route_dists[t_to] = new_dist_to
                            max_dist = new_max_val
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

        # Swap
        for t1 in range(truck_count):
            for i in range(len(routes[t1])):
                for t2 in range(t1+1, truck_count):
                    for j in range(len(routes[t2])):
                        new_route1 = routes[t1].copy()
                        new_route2 = routes[t2].copy()
                        new_route1[i] = routes[t2][j]
                        new_route2[j] = routes[t1][i]
                        new_dist1 = route_distance([0] + new_route1 + [0])
                        new_dist2 = route_distance([0] + new_route2 + [0])
                        other_max = max(route_dists[t] for t in range(truck_count) if t != t1 and t != t2)
                        new_max_val = max(new_dist1, new_dist2, other_max)
                        if new_max_val < max_dist - 1e-9:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            route_dists[t1] = new_dist1
                            route_dists[t2] = new_dist2
                            max_dist = new_max_val
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
                        other_max = max(route_dists[tt] for tt in range(truck_count) if tt != t)
                        new_max_val = max(new_dist, other_max)
                        if new_max_val < max_dist - 1e-9:
                            routes[t] = new_route
                            route_dists[t] = new_dist
                            max_dist = new_max_val
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Cross-route exchange
        for t1 in range(truck_count):
            for i in range(len(routes[t1])+1):
                for t2 in range(t1+1, truck_count):
                    for j in range(len(routes[t2])+1):
                        new_route1 = routes[t1][:i] + routes[t2][j:]
                        new_route2 = routes[t2][:j] + routes[t1][i:]
                        new_dist1 = route_distance([0] + new_route1 + [0]) if new_route1 else 0.0
                        new_dist2 = route_distance([0] + new_route2 + [0]) if new_route2 else 0.0
                        other_max = max(route_dists[t] for t in range(truck_count) if t != t1 and t != t2)
                        new_max_val = max(new_dist1, new_dist2, other_max)
                        if new_max_val < max_dist - 1e-9:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            route_dists[t1] = new_dist1
                            route_dists[t2] = new_dist2
                            max_dist = new_max_val
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        # After improvement, report
        if improved:
            report_best_vrp(get_full_routes(routes))

    return get_full_routes(best_routes)
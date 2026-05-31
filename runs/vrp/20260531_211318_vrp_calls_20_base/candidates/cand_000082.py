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
        for i in range(len(route) - 1):
            d += dist[route[i], route[i + 1]]
        return d

    # Initial construction: random order greedy insertion minimizing max distance
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
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists[:t]) + new_dist + sum(route_dists[t+1:])
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
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

    # Adaptive parameters
    max_iter = min(5000, 30 * n)
    T0 = best_max / 2.0
    T = T0
    removal_fraction_start = 0.3
    removal_fraction_end = 0.1

    # Operator weights (destroy: worst, random; repair: greedy, regret2)
    destroy_weights = [1.0, 1.0]
    repair_weights = [1.0, 1.0]
    destroy_history = [0.0, 0.0]  # cumulative score
    repair_history = [0.0, 0.0]
    decay_factor = 0.9  # weight decay per iteration

    def worst_removal(routes, dists, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if len(route) <= 2:
                continue
            base = dists[t]
            for i in range(1, len(route) - 1):
                prev = route[i - 1]
                nxt = route[i + 1]
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

    def random_removal(routes, dists, num_removals):
        all_customers = [c for r in routes for c in r[1:-1]]
        random.shuffle(all_customers)
        to_remove = set(all_customers[:num_removals])
        new_routes = []
        new_dists = []
        for route in routes:
            new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
            new_routes.append(new_route)
            new_dists.append(route_distance(new_route))
        return list(to_remove), new_routes, new_dists

    def greedy_repair(routes, dists, unassigned):
        routes = [list(r) for r in routes]
        dists = list(dists)
        unassigned = list(unassigned)
        current_max_local = max(dists)
        for cust in unassigned:
            best_truck = None
            best_pos = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            for t, route in enumerate(routes):
                old_dist = dists[t]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_distance(new_route)
                    delta_dist = new_dist - old_dist
                    new_max = max(current_max_local, new_dist)
                    new_total = sum(dists) + delta_dist
                    if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_truck = t
                        best_pos = pos
                        best_new_dist = new_dist
            routes[best_truck].insert(best_pos, cust)
            dists[best_truck] = best_new_dist
            if best_new_dist > current_max_local:
                current_max_local = best_new_dist
        return routes, dists

    def regret2_repair(routes, dists, unassigned):
        routes = [list(r) for r in routes]
        dists = list(dists)
        unassigned = list(unassigned)
        current_max_local = max(dists)
        while unassigned:
            best_info = None
            for cust in unassigned:
                best_max = float('inf')
                best_total = float('inf')
                best_truck = None
                best_pos = None
                second_best_max = float('inf')
                second_best_total = float('inf')
                for t, route in enumerate(routes):
                    old_dist = dists[t]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_distance(new_route)
                        delta_dist = new_dist - old_dist
                        new_max = max(current_max_local, new_dist)
                        new_total = sum(dists) + delta_dist
                        if new_max < best_max or (new_max == best_max and new_total < best_total):
                            second_best_max = best_max
                            second_best_total = best_total
                            best_max = new_max
                            best_total = new_total
                            best_truck = t
                            best_pos = pos
                            best_delta = delta_dist
                        elif new_max < second_best_max or (new_max == second_best_max and new_total < second_best_total):
                            second_best_max = new_max
                            second_best_total = new_total
                if second_best_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_best_max - best_max
                if best_info is None:
                    best_info = (regret, best_max, best_total, cust, best_truck, best_pos, best_delta)
                else:
                    if regret > best_info[0] or (regret == best_info[0] and (
                        best_max < best_info[1] or (best_max == best_info[1] and cust < best_info[3]))):
                        best_info = (regret, best_max, best_total, cust, best_truck, best_pos, best_delta)
            regret, best_max, best_total, cust, best_truck, best_pos, best_delta = best_info
            routes[best_truck].insert(best_pos, cust)
            dists[best_truck] += best_delta
            if dists[best_truck] > current_max_local:
                current_max_local = dists[best_truck]
            unassigned.remove(cust)
        return routes, dists

    for it in range(max_iter):
        # Adaptive removal fraction
        frac = removal_fraction_start + (removal_fraction_end - removal_fraction_start) * (it / max_iter)
        num_removals = max(1, int(frac * (n - 1)))

        # Select destroy operator via softmax on weights
        destroy_probs = [w / sum(destroy_weights) for w in destroy_weights]
        destroy_op = random.choices([0, 1], weights=destroy_probs)[0]
        # Select repair operator
        repair_probs = [w / sum(repair_weights) for w in repair_weights]
        repair_op = random.choices([0, 1], weights=repair_probs)[0]

        if destroy_op == 0:
            removed, partial, partial_dists = worst_removal(current_routes, current_dists, num_removals)
        else:
            removed, partial, partial_dists = random_removal(current_routes, current_dists, num_removals)

        if repair_op == 0:
            new_routes, new_dists = greedy_repair(partial, partial_dists, removed)
        else:
            new_routes, new_dists = regret2_repair(partial, partial_dists, removed)

        new_max = max(new_dists)
        new_total = sum(new_dists)
        delta = new_max - current_max
        accept = False
        if delta < 0 or (delta == 0 and new_total < current_total):
            accept = True
        elif random.random() < math.exp(-delta / max(T, 1e-9)):
            accept = True

        if accept:
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
                # Reward operators if improvement
                destroy_history[destroy_op] += 1.0
                repair_history[repair_op] += 1.0
            else:
                # Small reward for acceptance
                destroy_history[destroy_op] += 0.5
                repair_history[repair_op] += 0.5
        else:
            # No reward
            pass

        # Update weights using exponential moving average
        for i in range(2):
            destroy_weights[i] = decay_factor * destroy_weights[i] + (1 - decay_factor) * destroy_history[i]
            repair_weights[i] = decay_factor * repair_weights[i] + (1 - decay_factor) * repair_history[i]
        # Temperature decay
        T = T0 * (1 - it / max_iter)

    # Post-optimization: 2-opt improvement on best solution
    improved = True
    while improved:
        improved = False
        for t, route in enumerate(best_routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dists[t] - 1e-9:
                        old_other = sum(best_dists[:t]) + sum(best_dists[t+1:])
                        new_total = old_other + new_dist
                        new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
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

    return best_routes
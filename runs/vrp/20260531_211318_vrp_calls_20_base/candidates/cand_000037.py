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

    # Initial construction: random order greedy insertion minimizing max distance
    customers = list(range(1, n))
    random.shuffle(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_routes = routes[:t] + [new_route] + routes[t+1:]
                new_max = max(route_distance(r) for r in new_routes)
                new_total = sum(route_distance(r) for r in new_routes)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)

    current_routes = [list(r) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    # ALNS parameters
    max_iter = 50 * n
    removal_fraction = 0.3
    num_removals = max(1, int(removal_fraction * (n-1)))
    T0 = best_max / 2.0
    T = T0
    cooling_rate = 0.995

    # Adaptive operator weights
    n_destroy = 2
    n_repair = 2
    w_destroy = [1.0, 1.0]
    w_repair = [1.0, 1.0]
    segment_length = 100
    score_destroy = [0.0, 0.0]
    score_repair = [0.0, 0.0]
    usage_destroy = [0, 0]
    usage_repair = [0, 0]
    iteration_since_last_best = 0
    reheat_threshold = 0.2 * max_iter

    # Helper functions
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
            best_truck = None
            best_pos = None
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
                best_truck = None
                best_pos = None
                for t, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_routes = routes[:t] + [new_route] + routes[t+1:]
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
                    best_info = (regret, best_max, cust, best_truck, best_pos)
                else:
                    if regret > best_info[0] or (regret == best_info[0] and (best_max > best_info[1] or (best_max == best_info[1] and cust < best_info[2]))):
                        best_info = (regret, best_max, cust, best_truck, best_pos)
            regret, best_max_val, cust, best_truck, best_pos = best_info
            routes[best_truck].insert(best_pos, cust)
            unassigned.remove(cust)
        return routes

    destroy_ops = [worst_removal, random_removal]
    repair_ops = [greedy_repair, regret2_repair]

    for it in range(max_iter):
        # Adaptive selection
        total_w_d = sum(w_destroy)
        total_w_r = sum(w_repair)
        p_destroy = [w / total_w_d for w in w_destroy]
        p_repair = [w / total_w_r for w in w_repair]
        destroy_idx = random.choices([0,1], weights=p_destroy)[0]
        repair_idx = random.choices([0,1], weights=p_repair)[0]

        usage_destroy[destroy_idx] += 1
        usage_repair[repair_idx] += 1

        # Destroy
        removed, partial = destroy_ops[destroy_idx](current_routes, num_removals)
        # Repair
        new_routes = repair_ops[repair_idx](partial, removed)

        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)
        current_max = max(route_distance(r) for r in current_routes)
        current_total = sum(route_distance(r) for r in current_routes)

        delta = new_max - current_max
        accept = False
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_routes = [list(r) for r in new_routes]
            accept = True
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_distance(r) for r in best_routes)):
                best_max = new_max
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)
                iteration_since_last_best = 0
                # Score for new best: +5 for operators used
                score_destroy[destroy_idx] += 5.0
                score_repair[repair_idx] += 5.0
            else:
                iteration_since_last_best += 1
                # Score for accepted but not best: +1
                score_destroy[destroy_idx] += 1.0
                score_repair[repair_idx] += 1.0
        else:
            iteration_since_last_best += 1

        # Temperature update
        T = T0 * (cooling_rate ** (it + 1))
        # Reheat if stuck
        if iteration_since_last_best >= reheat_threshold and T < T0:
            T = T0
            iteration_since_last_best = 0

        # Update weights every segment
        if (it + 1) % segment_length == 0:
            for i in range(n_destroy):
                if usage_destroy[i] > 0:
                    w_destroy[i] = max(0.01, w_destroy[i] * 0.5 + 0.5 * (score_destroy[i] / usage_destroy[i]))
                score_destroy[i] = 0
                usage_destroy[i] = 0
            for i in range(n_repair):
                if usage_repair[i] > 0:
                    w_repair[i] = max(0.01, w_repair[i] * 0.5 + 0.5 * (score_repair[i] / usage_repair[i]))
                score_repair[i] = 0
                usage_repair[i] = 0

    return best_routes
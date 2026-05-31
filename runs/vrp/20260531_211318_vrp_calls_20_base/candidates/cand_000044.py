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
    removal_fraction = 0.35  # from cand_000023 for diversity
    num_removals = max(1, int(removal_fraction * (n-1)))
    T0 = best_max / 2.0
    T = T0

    # Adaptive operator weights (from cand_000018) plus usage counters for exploration bias
    destroy_weights = [1.0, 1.0]  # worst, random
    repair_weights = [1.0, 1.0]   # greedy, regret2
    destroy_usage = [1, 1]
    repair_usage = [1, 1]
    weight_increase = 0.1
    weight_decrease = 0.05
    min_weight = 0.1
    max_weight = 10.0

    # Helper functions for destroy and repair
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
            best_info = None  # (regret, best_max, cust, best_truck, best_pos)
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

    no_improve_iter = 0
    restart_threshold = int(0.2 * max_iter)

    for it in range(max_iter):
        # Random perturbation (5% chance from cand_000023)
        if random.random() < 0.05:
            all_custs = []
            for t, r in enumerate(current_routes):
                for i in range(1, len(r)-1):
                    all_custs.append((t, i, r[i]))
            if all_custs:
                t_old, pos_old, cust = random.choice(all_custs)
                route_old = current_routes[t_old]
                route_old.pop(pos_old)
                candidates = [x for x in range(truck_count) if x != t_old]
                if candidates:
                    t_new = random.choice(candidates)
                    route_new = current_routes[t_new]
                    best_pos_new = None
                    best_new_max = float('inf')
                    best_new_total = float('inf')
                    for pos in range(1, len(route_new)):
                        new_route = route_new[:pos] + [cust] + route_new[pos:]
                        new_routes = current_routes[:]
                        new_routes[t_old] = route_old
                        new_routes[t_new] = new_route
                        n_max = max(route_distance(r) for r in new_routes)
                        n_total = sum(route_distance(r) for r in new_routes)
                        if n_max < best_new_max or (n_max == best_new_max and n_total < best_new_total):
                            best_new_max = n_max
                            best_new_total = n_total
                            best_pos_new = pos
                    if best_pos_new is not None:
                        current_routes[t_new].insert(best_pos_new, cust)

        # Adaptive operator selection with exploration bias: combine weight and usage
        # Compute probabilities using weights scaled by (1 + 1/usage) to favor underused operators
        destroy_scores = []
        for w, u in zip(destroy_weights, destroy_usage):
            scaled = w * (1.0 + 1.0 / (u + 1))
            destroy_scores.append(scaled)
        total_d = sum(destroy_scores)
        probs_d = [s / total_d for s in destroy_scores]
        destroy_idx = random.choices([0, 1], weights=probs_d)[0]

        repair_scores = []
        for w, u in zip(repair_weights, repair_usage):
            scaled = w * (1.0 + 1.0 / (u + 1))
            repair_scores.append(scaled)
        total_r = sum(repair_scores)
        probs_r = [s / total_r for s in repair_scores]
        repair_idx = random.choices([0, 1], weights=probs_r)[0]

        # Destroy
        if destroy_idx == 0:
            removed, partial = worst_removal(current_routes, num_removals)
        else:
            removed, partial = random_removal(current_routes, num_removals)

        # Repair
        if repair_idx == 0:
            new_routes = greedy_repair(partial, removed)
        else:
            new_routes = regret2_repair(partial, removed)

        # Evaluate
        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)
        current_max = max(route_distance(r) for r in current_routes)
        current_total = sum(route_distance(r) for r in current_routes)

        delta = new_max - current_max
        accepted = False
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_routes = [list(r) for r in new_routes]
            accepted = True
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_distance(r) for r in best_routes)):
                best_max = new_max
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)
                no_improve_iter = 0
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1

        # Update operator weights and usage
        if accepted:
            destroy_weights[destroy_idx] = min(max_weight, destroy_weights[destroy_idx] + weight_increase)
            repair_weights[repair_idx] = min(max_weight, repair_weights[repair_idx] + weight_increase)
        else:
            destroy_weights[destroy_idx] = max(min_weight, destroy_weights[destroy_idx] - weight_decrease)
            repair_weights[repair_idx] = max(min_weight, repair_weights[repair_idx] - weight_decrease)
        destroy_usage[destroy_idx] += 1
        repair_usage[repair_idx] += 1

        # Temperature update
        T = T0 * (1 - it / max_iter)

        # Restart if stuck
        if no_improve_iter >= restart_threshold:
            large_removal_count = max(1, int(0.5 * (n-1)))
            removed, partial = random_removal(current_routes, large_removal_count)
            current_routes = greedy_repair(partial, removed)
            T = T0
            no_improve_iter = 0

    return best_routes
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
    T0 = best_max / 2.0
    T = T0

    # Sliding window for bandit selection
    window_size = 100
    decay = 0.9  # to age old records
    destroy_scores = [0.0, 0.0]
    repair_scores = [0.0, 0.0]
    destroy_counts = [1, 1]
    repair_counts = [1, 1]

    # Helper functions for destroy and repair (same as parents)
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

    # Local search: intra-route 2-opt and inter-route relocation
    def local_search(routes):
        # Intra-route 2-opt
        improved = True
        while improved:
            improved = False
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        if j-i == 1:
                            continue
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_routes = routes[:t] + [new_route] + routes[t+1:]
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max - 1e-9:
                            routes = [list(r) for r in new_routes]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        # Inter-route relocation: move customer to other route if reduces max
        for _ in range(n):  # limited iterations
            max_route_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
            max_route = routes[max_route_idx]
            best_improvement = 0.0
            best_move = None
            for i in range(1, len(max_route)-1):
                cust = max_route[i]
                # remove cust from max_route
                new_max_route = max_route[:i] + max_route[i+1:]
                for t2 in range(truck_count):
                    if t2 == max_route_idx:
                        continue
                    route2 = routes[t2]
                    for pos in range(1, len(route2)):
                        new_route2 = route2[:pos] + [cust] + route2[pos:]
                        new_routes = list(routes)
                        new_routes[max_route_idx] = new_max_route
                        new_routes[t2] = new_route2
                        new_max = max(route_distance(r) for r in new_routes)
                        improvement = best_max - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (max_route_idx, i, t2, pos, cust)
            if best_improvement > 1e-9:
                # apply move
                max_route_idx, i, t2, pos, cust = best_move
                route1 = routes[max_route_idx]
                route1.pop(i)
                routes[t2].insert(pos, cust)
                current_max = max(route_distance(r) for r in routes)
                if current_max < best_max - 1e-9:
                    best_max = current_max
                    best_routes[:] = [list(r) for r in routes]
                    report_best_vrp(best_routes)
            else:
                break
        return routes

    # Main loop
    for it in range(max_iter):
        # Dynamic removal fraction: linearly decrease from 0.4 to 0.2
        removal_fraction = 0.4 - 0.2 * (it / max_iter)
        num_removals = max(1, int(removal_fraction * (n-1)))

        # Bandit selection based on decaying scores
        total_d = sum(destroy_scores) + 1e-9
        destroy_probs = [(s + 0.1) / (total_d + 0.1 * 2) for s in destroy_scores]
        destroy_idx = random.choices([0, 1], weights=destroy_probs)[0]

        total_r = sum(repair_scores) + 1e-9
        repair_probs = [(s + 0.1) / (total_r + 0.1 * 2) for s in repair_scores]
        repair_idx = random.choices([0, 1], weights=repair_probs)[0]

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

        # Update bandit scores with decay
        for idx_list, scores, counts in [(destroy_scores, destroy_scores, destroy_counts), (repair_scores, repair_scores, repair_counts)]:
            for i in range(2):
                scores[i] *= decay
        # Increase score for selected operators if accepted
        if accepted:
            destroy_scores[destroy_idx] += 1.0
            repair_scores[repair_idx] += 1.0
        else:
            destroy_scores[destroy_idx] += 0.2  # small positive to avoid starvation
            repair_scores[repair_idx] += 0.2
        destroy_counts[destroy_idx] += 1
        repair_counts[repair_idx] += 1

        # Temperature update
        T = T0 * (1 - it / max_iter)

    # Post-optimization local search
    local_search(current_routes)
    current_max = max(route_distance(r) for r in current_routes)
    if current_max < best_max - 1e-9 or (abs(current_max - best_max) < 1e-9 and sum(route_distance(r) for r in current_routes) < sum(route_distance(r) for r in best_routes)):
        best_max = current_max
        best_routes = [list(r) for r in current_routes]
        report_best_vrp(best_routes)

    return best_routes
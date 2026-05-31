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

    # Adaptive operator selection
    # Weights for destroy (0: worst, 1: random) and repair (0: greedy, 1: regret2)
    destroy_weights = [1.0, 1.0]
    repair_weights = [1.0, 1.0]
    rho = 0.1  # reaction factor
    # Segment for removal fraction adjustment
    segment_size = max(1, int(max_iter / 20))
    iter_since_improvement = 0
    # Scores: 1 if new best, 0.5 if accepted, 0.1 if rejected

    # Destroy and repair functions identical to parent
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

    for it in range(max_iter):
        # Select operators using weights (roulette wheel)
        destroy_probs = [w / sum(destroy_weights) for w in destroy_weights]
        repair_probs = [w / sum(repair_weights) for w in repair_weights]
        destroy_op = random.choices([0, 1], weights=destroy_probs)[0]
        repair_op = random.choices([0, 1], weights=repair_probs)[0]

        # Destroy
        if destroy_op == 0:
            removed, partial = worst_removal(current_routes, num_removals)
        else:
            removed, partial = random_removal(current_routes, num_removals)

        # Repair
        if repair_op == 0:
            new_routes = greedy_repair(partial, removed)
        else:
            new_routes = regret2_repair(partial, removed)

        # Evaluate
        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)
        current_max = max(route_distance(r) for r in current_routes)
        current_total = sum(route_distance(r) for r in current_routes)

        delta = new_max - current_max
        accept = False
        score = 0.1  # base score
        if delta < 0 or (delta == 0 and new_total < current_total):
            accept = True
            score = 1.0 if (new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_distance(r) for r in best_routes))) else 0.5
        elif random.random() < math.exp(-delta / max(T, 1e-9)):
            accept = True
            score = 0.5
        else:
            score = 0.1

        if accept:
            current_routes = [list(r) for r in new_routes]
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_distance(r) for r in best_routes)):
                best_max = new_max
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)
                iter_since_improvement = 0
            else:
                iter_since_improvement += 1
        else:
            iter_since_improvement += 1

        # Update weights for the selected operators
        destroy_weights[destroy_op] = destroy_weights[destroy_op] * (1 - rho) + rho * score
        repair_weights[repair_op] = repair_weights[repair_op] * (1 - rho) + rho * score

        # Temperature update (linear cooling)
        T = T0 * (1 - it / max_iter)

        # Adaptive removal fraction adjustment every segment
        if (it + 1) % segment_size == 0:
            if iter_since_improvement >= segment_size:
                removal_fraction = min(0.5, removal_fraction + 0.05)
            else:
                removal_fraction = max(0.1, removal_fraction - 0.01)
            num_removals = max(1, int(removal_fraction * (n-1)))
            iter_since_improvement = 0

    return best_routes
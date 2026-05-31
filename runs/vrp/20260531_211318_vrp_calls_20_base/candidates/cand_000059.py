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

    # ALNS parameters (exploration-focused)
    max_iter = max(10 * n, 200)  # reduced iterations to avoid timeout
    removal_fraction = 0.3
    num_removals = max(1, int(removal_fraction * (n-1)))
    T0 = best_max * 1.5  # higher initial temperature
    T = T0

    # Operator usage counters
    destroy_weights = [1.0, 1.0, 1.0]  # worst, random, route-clear
    repair_weights = [1.0, 1.0]  # greedy, random

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
        if len(all_customers) == 0:
            return [], [list(r) for r in routes]
        random.shuffle(all_customers)
        to_remove = set(all_customers[:num_removals])
        new_routes = []
        for route in routes:
            new_routes.append([0] + [c for c in route[1:-1] if c not in to_remove] + [0])
        return list(to_remove), new_routes

    def route_clear_removal(routes, num_removals):
        # Remove all customers from one random route
        nonempty = [t for t, r in enumerate(routes) if len(r) > 2]
        if not nonempty:
            return [], [list(r) for r in routes]
        t = random.choice(nonempty)
        removed = routes[t][1:-1]
        new_routes = []
        for idx, r in enumerate(routes):
            if idx == t:
                new_routes.append([0, 0])
            else:
                new_routes.append(list(r))
        return list(removed), new_routes

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

    def random_repair(routes, unassigned):
        # Insert each customer at a random feasible position (any route, any position)
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        for cust in unassigned:
            possible = []
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    possible.append((t, pos))
            t, pos = random.choice(possible)
            routes[t].insert(pos, cust)
        return routes

    no_improve_iter = 0
    restart_threshold = int(0.05 * max_iter)  # frequent restarts

    for it in range(max_iter):
        # Select destroy operator using roulette wheel with exploration (inverse weight)
        total_d = sum(destroy_weights)
        destroy_probs = [total_d / (w + 1e-9) for w in destroy_weights]
        destroy_probs = [p / sum(destroy_probs) for p in destroy_probs]
        destroy_op = random.choices([0, 1, 2], weights=destroy_probs)[0]

        # Destroy
        if destroy_op == 0:
            removed, partial = worst_removal(current_routes, num_removals)
        elif destroy_op == 1:
            removed, partial = random_removal(current_routes, num_removals)
        else:
            removed, partial = route_clear_removal(current_routes, num_removals)

        # Select repair operator (also with exploration)
        total_r = sum(repair_weights)
        repair_probs = [total_r / (w + 1e-9) for w in repair_weights]
        repair_probs = [p / sum(repair_probs) for p in repair_probs]
        repair_op = random.choices([0, 1], weights=repair_probs)[0]

        # Repair
        if repair_op == 0:
            new_routes = greedy_repair(partial, removed)
        else:
            new_routes = random_repair(partial, removed)

        # Evaluate
        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)
        current_max = max(route_distance(r) for r in current_routes)
        current_total = sum(route_distance(r) for r in current_routes)

        delta = new_max - current_max
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_routes = [list(r) for r in new_routes]
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_distance(r) for r in best_routes)):
                best_max = new_max
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)
                no_improve_iter = 0
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1

        # Update operator weights (increase usage of selected operators)
        destroy_weights[destroy_op] += 0.1
        repair_weights[repair_op] += 0.1

        # Temperature update (linear cooling)
        T = T0 * (1 - it / max_iter)

        # Restart if stuck: build completely new random solution
        if no_improve_iter >= restart_threshold:
            customers = list(range(1, n))
            random.shuffle(customers)
            new_routes = [[0, 0] for _ in range(truck_count)]
            for cust in customers:
                best_truck = None
                best_pos = None
                best_max_tmp = float('inf')
                best_total_tmp = float('inf')
                for t, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        candidate_route = route[:pos] + [cust] + route[pos:]
                        candidate_routes = new_routes[:t] + [candidate_route] + new_routes[t+1:]
                        cand_max = max(route_distance(r) for r in candidate_routes)
                        cand_total = sum(route_distance(r) for r in candidate_routes)
                        if cand_max < best_max_tmp or (cand_max == best_max_tmp and cand_total < best_total_tmp):
                            best_max_tmp = cand_max
                            best_total_tmp = cand_total
                            best_truck = t
                            best_pos = pos
                new_routes[best_truck].insert(best_pos, cust)
            current_routes = [list(r) for r in new_routes]
            T = T0
            # Reset weights slightly to avoid biasing
            destroy_weights = [1.0, 1.0, 1.0]
            repair_weights = [1.0, 1.0]
            no_improve_iter = 0

    return best_routes
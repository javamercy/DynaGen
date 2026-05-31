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

    # Initial construction: greedy with random order (same as parent)
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
    best_total = sum(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    # ALNS parameters - more exploration
    max_iter = 20 * n
    removal_fraction = 0.2
    num_removals = max(1, int(removal_fraction * (n-1)))
    T0 = best_max  # higher initial temperature
    T = T0

    # Operator controls: destroy: 0=worst, 1=random; repair: 0=greedy, 1=regret2
    destroy_usage = [1, 1]
    repair_usage = [1, 1]
    destroy_weights = [1.0, 1.0]
    repair_weights = [1.0, 1.0]
    destroy_scores = [0.0, 0.0]
    repair_scores = [0.0, 0.0]
    # Track success for adaptive weights (reactive)
    def update_weights(usage, scores, weights, decay=0.5):
        total_usage = sum(usage) if sum(usage) > 0 else 1
        for i in range(len(weights)):
            if usage[i] > 0:
                scores[i] = (1 - decay) * scores[i] + decay * (scores[i] / usage[i])
            else:
                scores[i] = 0.001
        total_score = sum(scores)
        if total_score > 0:
            for i in range(len(weights)):
                weights[i] = scores[i] / total_score
        else:
            for i in range(len(weights)):
                weights[i] = 1.0 / len(weights)

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
            # Compute regret-2 for each unassigned customer
            cust_records = []
            for cust in unassigned:
                insertion_costs = []
                for t, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_routes = routes[:t] + [new_route] + routes[t+1:]
                        new_max_val = max(route_distance(r) for r in new_routes)
                        insertion_costs.append((new_max_val, t, pos))
                # Sort by cost ascending
                insertion_costs.sort(key=lambda x: x[0])
                best_cost = insertion_costs[0][0]
                second_best_cost = insertion_costs[1][0] if len(insertion_costs) > 1 else best_cost
                regret = second_best_cost - best_cost
                cust_records.append((-regret, cust, best_cost, insertion_costs[0][1], insertion_costs[0][2]))
            # Sort by regret descending
            cust_records.sort(key=lambda x: x[0])
            _, cust, _, t, pos = cust_records[0]
            routes[t].insert(pos, cust)
            unassigned.remove(cust)
        return routes

    def balance_local_search(routes):
        # Try to move a customer from max-distance route to another to reduce max
        max_route_idx = max(range(len(routes)), key=lambda i: route_distance(routes[i]))
        max_route = routes[max_route_idx]
        if len(max_route) <= 2:
            return routes
        best_routes = None
        best_max = route_distance(routes[max_route_idx])
        best_total = sum(route_distance(r) for r in routes)
        for i in range(1, len(max_route)-1):
            cust = max_route[i]
            new_max_route = [0] + [c for c in max_route[1:-1] if c != cust] + [0]
            if len(new_max_route) <= 2:
                new_max_route = [0, 0]
            for t in range(len(routes)):
                if t == max_route_idx:
                    continue
                route = routes[t]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    candidate_routes = list(routes)
                    candidate_routes[max_route_idx] = new_max_route
                    candidate_routes[t] = new_route
                    new_max_val = max(route_distance(r) for r in candidate_routes)
                    new_total_val = sum(route_distance(r) for r in candidate_routes)
                    if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                        best_max = new_max_val
                        best_total = new_total_val
                        best_routes = [list(r) for r in candidate_routes]
        if best_routes is not None:
            return best_routes
        return routes

    no_improve_iter = 0
    restart_threshold = int(0.1 * max_iter)
    balance_freq = max(1, int(max_iter / 10))
    shuffle_freq = max(1, int(max_iter / 20))  # periodic random perturbation

    for it in range(max_iter):
        # Adaptive selection of destroy operator
        total_destroy_weights = sum(destroy_weights)
        destroy_probs = [w / total_destroy_weights for w in destroy_weights]
        destroy_op = random.choices([0, 1], weights=destroy_probs)[0]

        # Adaptive selection of repair operator
        total_repair_weights = sum(repair_weights)
        repair_probs = [w / total_repair_weights for w in repair_weights]
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
        if delta < 0 or (delta == 0 and new_total < current_total):
            accept = True
        elif random.random() < math.exp(-delta / max(T, 1e-9)):
            accept = True

        if accept:
            current_routes = [list(r) for r in new_routes]
            # Update scores for operators that led to acceptance
            destroy_scores[destroy_op] += 1.0
            repair_scores[repair_op] += 1.0
            destroy_usage[destroy_op] += 1
            repair_usage[repair_op] += 1
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_distance(r) for r in best_routes)):
                best_max = new_max
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)
                no_improve_iter = 0
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1

        # Update weights every 5 iterations
        if it % 5 == 0:
            update_weights(destroy_usage, destroy_scores, destroy_weights)
            update_weights(repair_usage, repair_scores, repair_weights)

        # Temperature update (slow cooling)
        T = T0 * (1 - it / max_iter)

        # Balance local search periodically
        if it % balance_freq == 0:
            current_routes = balance_local_search(current_routes)
            current_max = max(route_distance(r) for r in current_routes)
            current_total = sum(route_distance(r) for r in current_routes)
            if current_max < best_max - 1e-9 or (abs(current_max - best_max) < 1e-9 and current_total < sum(route_distance(r) for r in best_routes)):
                best_max = current_max
                best_routes = [list(r) for r in current_routes]
                report_best_vrp(best_routes)
                no_improve_iter = 0

        # Shuffle perturbation (diverse search)
        if it % shuffle_freq == 0 and it > 0:
            # Remove a random subset (30% of customers) and reinsert greedily with some randomness
            num_shuffle = max(1, int(0.3 * (n-1)))
            removed, partial = random_removal(current_routes, num_shuffle)
            # Use greedy with random tie-breaking? Actually just use greedy repair
            current_routes = greedy_repair(partial, removed)
            T = T0  # Reset temperature for diversification

        # Restart if stuck
        if no_improve_iter >= restart_threshold:
            large_removal_count = max(1, int(0.5 * (n-1)))  # Remove 50%
            removed, partial = random_removal(current_routes, large_removal_count)
            current_routes = greedy_repair(partial, removed)
            T = T0
            no_improve_iter = 0
            # Reset operator scores for more exploration
            destroy_scores = [0.0, 0.0]
            repair_scores = [0.0, 0.0]
            destroy_usage = [1, 1]
            repair_usage = [1, 1]

    return best_routes
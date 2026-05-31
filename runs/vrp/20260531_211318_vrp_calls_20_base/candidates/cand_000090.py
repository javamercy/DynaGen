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

    # Initial construction: random order greedy insertion (minimize max then total)
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

    # ALNS parameters
    max_iter = max(100, 3 * n)
    removal_fraction = 0.15
    num_removals = max(1, int(removal_fraction * (n-1)))
    # Record-to-record deviation (5% of best)
    deviation = 0.05

    # Adaptive operator weights (scores)
    destroy_scores = [1, 1]  # worst, random
    destroy_usage = [1, 1]
    score_best = 3
    score_accepted = 1
    score_rejected = 0

    def worst_removal(routes, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if len(route) <= 2:
                continue
            base = route_distance(route)
            for i in range(1, len(route)-1):
                prev = route[i-1]
                nxt = route[i+1]
                contrib = dist[prev, route[i]] + dist[route[i], nxt] - dist[prev, nxt]
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

    no_improve_iter = 0
    restart_threshold = int(0.2 * max_iter)

    for it in range(max_iter):
        # Select destroy operator via roulette wheel based on scores
        total_score = sum(destroy_scores)
        probs = [s / total_score for s in destroy_scores]
        destroy_op = random.choices([0, 1], weights=probs)[0]

        # Destroy
        if destroy_op == 0:
            removed, partial = worst_removal(current_routes, num_removals)
        else:
            removed, partial = random_removal(current_routes, num_removals)

        # Repair (greedy)
        new_routes = greedy_repair(partial, removed)

        # Evaluate
        new_max = max(route_distance(r) for r in new_routes)
        new_total = sum(route_distance(r) for r in new_routes)

        # Record-to-record acceptance: accept if new_max <= best_max * (1 + deviation)
        # Or if equal on max and better on total, also accept.
        accepted = False
        if new_max < best_max * (1 + deviation) - 1e-9:
            accepted = True
        elif abs(new_max - best_max * (1 + deviation)) < 1e-9 and new_total < best_total:
            accepted = True

        if accepted:
            current_routes = [list(r) for r in new_routes]
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes]
                report_best_vrp(best_routes)
                no_improve_iter = 0
                destroy_scores[destroy_op] += score_best
            else:
                no_improve_iter += 1
                destroy_scores[destroy_op] += score_accepted
        else:
            no_improve_iter += 1
            destroy_scores[destroy_op] += score_rejected

        # Restart if stuck
        if no_improve_iter >= restart_threshold:
            large_removal_count = max(1, int(0.5 * (n-1)))
            removed, partial = random_removal(current_routes, large_removal_count)
            current_routes = greedy_repair(partial, removed)
            no_improve_iter = 0

    return best_routes
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
    max_iter = 20 * n
    removal_fraction = 0.25
    num_removals = max(1, int(removal_fraction * (n-1)))
    T0 = best_max / 2.0
    T = T0

    # Operator usage counters
    destroy_usage = [1, 1]
    repair_usage = [1]

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

    def intra_route_2opt(routes):
        improved = True
        max_local_iters = 10 * n
        it = 0
        while improved and it < max_local_iters:
            improved = False
            for t, route in enumerate(routes):
                if len(route) <= 4:
                    continue
                best_delta = 0
                best_i = -1
                best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        if j-i == 1:
                            continue
                        # calculate delta from reversing segment (i+1, j)
                        a, b, c, d = route[i-1], route[i], route[j], route[j+1]
                        old = dist[a, b] + dist[c, d]
                        new = dist[a, c] + dist[b, d]
                        delta = new - old
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
                if best_delta < -1e-9:
                    # apply best move
                    routes[t][best_i:best_j+1] = list(reversed(routes[t][best_i:best_j+1]))
                    improved = True
            it += 1
        return routes

    no_improve_iter = 0
    restart_threshold = int(0.1 * max_iter)

    for it in range(max_iter):
        # Operator selection with exploration bias
        total_destroy = sum(destroy_usage)
        destroy_probs = [total_destroy / u for u in destroy_usage]
        destroy_probs = [p / sum(destroy_probs) for p in destroy_probs]
        destroy_op = random.choices([0, 1], weights=destroy_probs)[0]

        # Destroy
        if destroy_op == 0:
            removed, partial = worst_removal(current_routes, num_removals)
        else:
            removed, partial = random_removal(current_routes, num_removals)

        # Repair (greedy only)
        new_routes = greedy_repair(partial, removed)

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
                # Apply intra-route 2-opt to improve further
                improved_routes = intra_route_2opt([list(r) for r in current_routes])
                improved_max = max(route_distance(r) for r in improved_routes)
                improved_total = sum(route_distance(r) for r in improved_routes)
                if improved_max < new_max - 1e-9 or (abs(improved_max - new_max) < 1e-9 and improved_total < new_total):
                    current_routes = [list(r) for r in improved_routes]
                    new_max = improved_max
                    new_total = improved_total
                if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < sum(route_distance(r) for r in best_routes)):
                    best_max = new_max
                    best_routes = [list(r) for r in current_routes]
                    report_best_vrp(best_routes)
                    no_improve_iter = 0
                else:
                    no_improve_iter += 1
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1

        # Update operator usage
        destroy_usage[destroy_op] += 1

        # Temperature update
        T = T0 * (1 - it / max_iter)

        # Restart if stuck
        if no_improve_iter >= restart_threshold:
            large_removal_count = max(1, int(0.5 * (n-1)))
            removed, partial = random_removal(current_routes, large_removal_count)
            current_routes = greedy_repair(partial, removed)
            T = T0
            no_improve_iter = 0

    # Final inter-route improvement
    def inter_route_improve(routes):
        improved = True
        max_iters = 5 * n
        it = 0
        while improved and it < max_iters:
            improved = False
            best_move = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            # Try relocate each customer to each position in another route
            for t1, route1 in enumerate(routes):
                for i in range(1, len(route1)-1):
                    cust = route1[i]
                    # Remove cust from route1
                    new_route1 = route1[:i] + route1[i+1:]
                    # Try inserting into all other routes
                    for t2, route2 in enumerate(routes):
                        if t2 == t1:
                            continue
                        for j in range(1, len(route2)):
                            new_route2 = route2[:j] + [cust] + route2[j:]
                            new_routes = routes[:]
                            new_routes[t1] = new_route1 if len(new_route1) >= 2 else [0,0]
                            new_routes[t2] = new_route2
                            new_max = max(route_distance(r) for r in new_routes)
                            new_total = sum(route_distance(r) for r in new_routes)
                            if new_max < best_new_max - 1e-9 or (abs(new_max - best_new_max) < 1e-9 and new_total < best_new_total):
                                best_new_max = new_max
                                best_new_total = new_total
                                best_move = (t1, i, cust, t2, j)
            # Try exchange customers between routes
            for t1, route1 in enumerate(routes):
                for i in range(1, len(route1)-1):
                    cust1 = route1[i]
                    for t2 in range(t1+1, len(routes)):
                        route2 = routes[t2]
                        for j in range(1, len(route2)-1):
                            cust2 = route2[j]
                            # Swap
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            new_routes = routes[:]
                            new_routes[t1] = new_route1
                            new_routes[t2] = new_route2
                            new_max = max(route_distance(r) for r in new_routes)
                            new_total = sum(route_distance(r) for r in new_routes)
                            if new_max < best_new_max - 1e-9 or (abs(new_max - best_new_max) < 1e-9 and new_total < best_new_total):
                                best_new_max = new_max
                                best_new_total = new_total
                                best_move = ('exchange', t1, i, cust1, t2, j, cust2)
            if best_move is not None and best_new_max < max(route_distance(r) for r in routes) - 1e-9:
                # Apply move
                if best_move[0] == 'exchange':
                    _, t1, i, cust1, t2, j, cust2 = best_move
                    routes[t1][i] = cust2
                    routes[t2][j] = cust1
                else:
                    t1, i, cust, t2, j = best_move
                    new_route1 = routes[t1][:i] + routes[t1][i+1:]
                    if len(new_route1) == 1:
                        new_route1 = [0,0]
                    routes[t1] = new_route1
                    routes[t2] = routes[t2][:j] + [cust] + routes[t2][j:]
                improved = True
            it += 1
        return routes

    best_routes = inter_route_improve([list(r) for r in best_routes])
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    return best_routes
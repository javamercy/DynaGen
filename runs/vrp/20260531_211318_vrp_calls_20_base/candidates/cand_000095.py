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

    # Random initial construction: assign each customer to a random truck at best position
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

    # ALNS parameters
    max_iter = min(5000, 30 * n)
    removal_fraction = 0.3  # increase removal size for more shake
    num_removals = max(2, int(removal_fraction * (n - 1)))
    T0 = best_max * 0.8
    T = T0
    no_improve = 0
    restart_limit = 100

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

    def ruin_recreate_removal(routes, dists, num_removals):
        # pick a random seed customer and remove its k nearest neighbors
        all_customers = [c for r in routes for c in r[1:-1]]
        if not all_customers:
            return [], [list(r) for r in routes], list(dists)
        seed = random.choice(all_customers)
        distances_from_seed = [(c, dist[seed][c]) for c in all_customers if c != seed]
        distances_from_seed.sort(key=lambda x: x[1])
        k = min(num_removals, len(distances_from_seed) + 1)
        to_remove = {seed}
        for i in range(k - 1):
            to_remove.add(distances_from_seed[i][0])
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
                best_max_val = float('inf')
                best_total_val = float('inf')
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
                        if new_max < best_max_val or (new_max == best_max_val and new_total < best_total_val):
                            second_best_max = best_max_val
                            second_best_total = best_total_val
                            best_max_val = new_max
                            best_total_val = new_total
                            best_truck = t
                            best_pos = pos
                            best_delta = delta_dist
                        elif new_max < second_best_max or (new_max == second_best_max and new_total < second_best_total):
                            second_best_max = new_max
                            second_best_total = new_total
                if second_best_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_best_max - best_max_val
                if best_info is None:
                    best_info = (regret, best_max_val, best_total_val, cust, best_truck, best_pos, best_delta)
                else:
                    if regret > best_info[0] or (regret == best_info[0] and (
                        best_max_val < best_info[1] or (best_max_val == best_info[1] and cust < best_info[3]))):
                        best_info = (regret, best_max_val, best_total_val, cust, best_truck, best_pos, best_delta)
            regret, best_max_val, best_total_val, cust, best_truck, best_pos, best_delta = best_info
            routes[best_truck].insert(best_pos, cust)
            dists[best_truck] += best_delta
            if dists[best_truck] > current_max_local:
                current_max_local = dists[best_truck]
            unassigned.remove(cust)
        return routes, dists

    # Main ALNS loop with restart
    for it in range(max_iter):
        # Diversify: choose destroy operator with higher random probability
        rnd = random.random()
        if rnd < 0.5:
            destroy_op = 2  # ruin and recreate
        elif rnd < 0.8:
            destroy_op = 1  # random
        else:
            destroy_op = 0  # worst

        repair_op = random.choice([0, 1])

        if destroy_op == 0:
            removed, partial, partial_dists = worst_removal(current_routes, current_dists, num_removals)
        elif destroy_op == 1:
            removed, partial, partial_dists = random_removal(current_routes, current_dists, num_removals)
        else:
            removed, partial, partial_dists = ruin_recreate_removal(current_routes, current_dists, num_removals)

        if repair_op == 0:
            new_routes, new_dists = greedy_repair(partial, partial_dists, removed)
        else:
            new_routes, new_dists = regret2_repair(partial, partial_dists, removed)

        new_max = max(new_dists)
        new_total = sum(new_dists)
        delta = new_max - current_max
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
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
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        T = T0 * (1 - it / max_iter)

        # Restart if stuck
        if no_improve >= restart_limit and it < max_iter - 10:
            # Generate a new random solution from scratch
            customers = list(range(1, n))
            random.shuffle(customers)
            new_routes = [[0, 0] for _ in range(truck_count)]
            new_dists = [0.0] * truck_count
            for cust in customers:
                best_t = random.randrange(truck_count)
                best_p = random.randrange(1, len(new_routes[best_t]))
                new_route = new_routes[best_t][:best_p] + [cust] + new_routes[best_t][best_p:]
                new_routes[best_t] = new_route
                new_dists[best_t] = route_distance(new_route)
            current_routes = new_routes
            current_dists = new_dists
            current_max = max(current_dists)
            current_total = sum(current_dists)
            T = T0 * 1.5  # increase temperature
            no_improve = 0

    # Final 2-opt improvement on best solution
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
                        new_total_val = old_other + new_dist
                        new_max_val = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                        if new_max_val < best_max - 1e-9 or (abs(new_max_val - best_max) < 1e-9 and new_total_val < best_total):
                            best_routes[t] = new_route
                            best_dists[t] = new_dist
                            best_max = new_max_val
                            best_total = new_total_val
                            report_best_vrp(best_routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    return best_routes
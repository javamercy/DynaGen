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

    # Initial construction: farthest-first greedy insertion minimizing max distance
    customers = sorted(range(1, n), key=lambda c: -dist[0, c])
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
    removal_fraction = 0.25
    num_removals = max(1, int(removal_fraction * (n - 1)))
    T0 = best_max / 2.0
    T = T0

    # Adaptive operator scores
    destroy_scores = [1, 1]  # worst, random
    score_best = 3
    score_accepted = 1
    score_rejected = 0

    def worst_removal(routes, dists, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if len(route) <= 2:
                continue
            for i in range(1, len(route) - 1):
                prev = route[i - 1]
                nxt = route[i + 1]
                contrib = dist[prev, route[i]] + dist[route[i], nxt] - dist[prev, nxt]
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
                        delta = new_dist - old_dist
                        new_max = max(current_max_local, new_dist)
                        new_total = sum(dists) + delta
                        if new_max < best_max_val or (new_max == best_max_val and new_total < best_total_val):
                            second_best_max = best_max_val
                            second_best_total = best_total_val
                            best_max_val = new_max
                            best_total_val = new_total
                            best_truck = t
                            best_pos = pos
                            best_delta = delta
                        elif new_max < second_best_max or (new_max == second_best_max and new_total < second_best_total):
                            second_best_max = new_max
                            second_best_total = new_total
                if second_best_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_best_max - best_max_val
                if best_info is None or regret > best_info[0]:
                    best_info = (regret, best_max_val, best_total_val, cust, best_truck, best_pos, best_delta)
            _, _, _, cust, best_truck, best_pos, best_delta = best_info
            routes[best_truck].insert(best_pos, cust)
            dists[best_truck] += best_delta
            if dists[best_truck] > current_max_local:
                current_max_local = dists[best_truck]
            unassigned.remove(cust)
        return routes, dists

    no_improve_iter = 0
    restart_threshold = int(0.2 * max_iter)

    for it in range(max_iter):
        # Select destroy operator via roulette wheel
        total_score = sum(destroy_scores)
        probs = [s / total_score for s in destroy_scores]
        destroy_op = random.choices([0, 1], weights=probs)[0]

        if destroy_op == 0:
            removed, partial, partial_dists = worst_removal(current_routes, current_dists, num_removals)
        else:
            removed, partial, partial_dists = random_removal(current_routes, current_dists, num_removals)

        new_routes, new_dists = regret2_repair(partial, partial_dists, removed)

        new_max = max(new_dists)
        new_total = sum(new_dists)
        delta = new_max - current_max
        accepted = False
        if delta < 0 or (delta == 0 and new_total < current_total) or random.random() < math.exp(-delta / max(T, 1e-9)):
            accepted = True
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
                no_improve_iter = 0
                destroy_scores[destroy_op] += score_best
            else:
                no_improve_iter += 1
                destroy_scores[destroy_op] += score_accepted
        else:
            no_improve_iter += 1
            destroy_scores[destroy_op] += score_rejected

        T = T0 * (1 - it / max_iter)

        # Restart if stuck
        if no_improve_iter >= restart_threshold:
            large_removal_count = max(1, int(0.5 * (n - 1)))
            removed, partial, partial_dists = random_removal(current_routes, current_dists, large_removal_count)
            # Quick greedy repair for restart
            new_routes, new_dists = regret2_repair(partial, partial_dists, removed)
            current_routes = [list(r) for r in new_routes]
            current_dists = list(new_dists)
            current_max = max(current_dists)
            current_total = sum(current_dists)
            T = T0
            no_improve_iter = 0

    # Post-optimization: 2-opt on best solution
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
                        new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                        new_total = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
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

    # Intensification: focus on longest route
    for iteration in range(min(100, 5 * n)):
        t_max = max(range(truck_count), key=lambda t: best_dists[t])
        route_max = best_routes[t_max]
        if len(route_max) <= 2:
            break
        # Select up to 3 customers with highest removal contribution
        contribs = []
        for i in range(1, len(route_max) - 1):
            prev = route_max[i-1]
            nxt = route_max[i+1]
            contrib = dist[prev, route_max[i]] + dist[route_max[i], nxt] - dist[prev, nxt]
            contribs.append((contrib, i, route_max[i]))
        contribs.sort(key=lambda x: -x[0])
        k = random.randint(1, min(3, len(route_max)-2))
        to_remove = set()
        for _, i, cust in contribs[:k]:
            to_remove.add(cust)
        # Remove from best_routes temporarily
        saved_routes = [list(r) for r in best_routes]
        saved_dists = list(best_dists)
        new_route_max = [0] + [c for c in route_max[1:-1] if c not in to_remove] + [0]
        new_dist_max = route_distance(new_route_max)
        best_routes[t_max] = new_route_max
        best_dists[t_max] = new_dist_max
        # Reinsert using regret2_repair
        removed_list = list(to_remove)
        result_routes, result_dists = regret2_repair(best_routes, best_dists, removed_list)
        new_best_max = max(result_dists)
        new_best_total = sum(result_dists)
        if new_best_max < best_max - 1e-9 or (abs(new_best_max - best_max) < 1e-9 and new_best_total < best_total):
            best_routes = [list(r) for r in result_routes]
            best_dists = list(result_dists)
            best_max = new_best_max
            best_total = new_best_total
            report_best_vrp(best_routes)
        else:
            best_routes = saved_routes
            best_dists = saved_dists
        # Apply 2-opt on each route after change
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
                            new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                            new_total = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
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
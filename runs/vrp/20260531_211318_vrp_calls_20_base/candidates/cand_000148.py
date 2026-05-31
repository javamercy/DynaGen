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

    def insertion_delta(route, pos, cust):
        prev = route[pos-1]
        nxt = route[pos]
        return dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]

    def removal_delta(route, pos):
        prev = route[pos-1]
        nxt = route[pos+1]
        return dist[prev, route[pos]] + dist[route[pos], nxt] - dist[prev, nxt]

    # 2-opt improvement on a single route (first improvement, limited passes)
    def two_opt_route(route):
        if len(route) <= 3:
            return route
        improved = True
        max_passes = len(route)  # assure termination
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    delta = - dist[route[i-1], route[i]] - dist[route[j], route[j+1]] + dist[route[i-1], route[j]] + dist[route[i], route[j+1]]
                    if delta < -1e-9:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        break
                if improved:
                    break
        return route

    # Farthest-first initial construction
    customers = sorted(range(1, n), key=lambda c: -dist[0][c])
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_dist = route_dists[t] + insertion_delta(route, pos, cust)
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists) + insertion_delta(route, pos, cust)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        route = routes[best_truck]
        routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
        route_dists[best_truck] += insertion_delta(route, best_pos, cust)

    current_routes = [list(r) for r in routes]
    current_dists = list(route_dists)
    current_max = max(current_dists)
    current_total = sum(current_dists)
    best_routes = [list(r) for r in routes]
    best_dists = list(route_dists)
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    # Adaptive scores for 3 destroy and 3 repair
    destroy_scores = [1.0, 1.0, 1.0]
    repair_scores = [1.0, 1.0, 1.0]
    score_best = 3.0
    score_accepted = 1.0
    score_rejected = 0.0

    max_iter = min(3000, 20 * n)
    removal_fraction_start = 0.3
    removal_fraction_end = 0.1
    beta_start = 0.05
    beta_end = 0.005

    no_improve_iter = 0
    last_best_iter = 0

    for it in range(max_iter):
        frac = removal_fraction_start + (removal_fraction_end - removal_fraction_start) * (it / max_iter)
        num_removals = max(1, int(frac * (n - 1)))
        beta = beta_start + (beta_end - beta_start) * (it / max_iter)

        # Select destroy
        total_d = sum(destroy_scores)
        destroy_probs = [s / total_d for s in destroy_scores]
        destroy_op = random.choices([0, 1, 2], weights=destroy_probs)[0]

        # Destroy
        if destroy_op == 0:  # worst removal (by delta impact)
            all_contribs = []
            for t, route in enumerate(current_routes):
                if len(route) <= 2:
                    continue
                for pos in range(1, len(route)-1):
                    contrib = removal_delta(route, pos)
                    all_contribs.append((contrib, t, pos, route[pos]))
            all_contribs.sort(key=lambda x: (-x[0], x[3]))
            to_remove = set()
            for _, t, pos, cust in all_contribs[:num_removals]:
                to_remove.add(cust)
            new_routes = []
            new_dists = []
            for t, route in enumerate(current_routes):
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
        elif destroy_op == 1:  # random removal
            all_customers = [c for r in current_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:num_removals])
            new_routes = []
            new_dists = []
            for route in current_routes:
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
        else:  # worst-on-max-route: remove from the route with highest distance
            t_max = max(range(truck_count), key=lambda t: current_dists[t])
            route_max = current_routes[t_max]
            if len(route_max) <= 2:
                to_remove = set()
            else:
                contribs = []
                for pos in range(1, len(route_max)-1):
                    contrib = removal_delta(route_max, pos)
                    contribs.append((contrib, pos, route_max[pos]))
                contribs.sort(key=lambda x: (-x[0], x[2]))
                to_remove = set(c for _, _, c in contribs[:min(num_removals, len(contribs))])
            new_routes = []
            new_dists = []
            for t, route in enumerate(current_routes):
                if t == t_max:
                    new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                else:
                    new_route = [0] + [c for c in route[1:-1]] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
        removed = list(to_remove)

        # Select repair
        total_r = sum(repair_scores)
        repair_probs = [s / total_r for s in repair_scores]
        repair_op = random.choices([0, 1, 2], weights=repair_probs)[0]

        # Repair
        if repair_op == 0:  # greedy (min total distance)
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(removed)
            current_max_repair = max(dists_repair)
            for cust in unassigned:
                best_truck = None
                best_pos = None
                best_new_max = float('inf')
                best_new_total = float('inf')
                for t, route in enumerate(routes_repair):
                    old_dist = dists_repair[t]
                    for pos in range(1, len(route)):
                        delta = insertion_delta(route, pos, cust)
                        new_dist = old_dist + delta
                        new_max = max(current_max_repair, new_dist)
                        new_total = sum(dists_repair) + delta
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_truck = t
                            best_pos = pos
                routes_repair[best_truck] = routes_repair[best_truck][:best_pos] + [cust] + routes_repair[best_truck][best_pos:]
                dists_repair[best_truck] += insertion_delta(routes_repair[best_truck], best_pos, cust)
                if dists_repair[best_truck] > current_max_repair:
                    current_max_repair = dists_repair[best_truck]
            new_routes_final = routes_repair
            new_dists_final = dists_repair
        elif repair_op == 1:  # regret-2 (by max)
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(removed)
            current_max_repair = max(dists_repair)
            while unassigned:
                best_info = None
                for cust in unassigned:
                    best_max_val = float('inf')
                    best_total_val = float('inf')
                    best_truck = None
                    best_pos = None
                    second_best_max = float('inf')
                    second_best_total = float('inf')
                    for t, route in enumerate(routes_repair):
                        old_dist = dists_repair[t]
                        for pos in range(1, len(route)):
                            delta = insertion_delta(route, pos, cust)
                            new_dist = old_dist + delta
                            new_max = max(current_max_repair, new_dist)
                            new_total = sum(dists_repair) + delta
                            if new_max < best_max_val or (new_max == best_max_val and new_total < best_total_val):
                                second_best_max = best_max_val
                                second_best_total = best_total_val
                                best_max_val = new_max
                                best_total_val = new_total
                                best_truck = t
                                best_pos = pos
                            elif new_max < second_best_max or (new_max == second_best_max and new_total < second_best_total):
                                second_best_max = new_max
                                second_best_total = new_total
                    regret = (second_best_max - best_max_val) if second_best_max != float('inf') else float('inf')
                    if best_info is None or regret > best_info[0] or (regret == best_info[0] and (best_max_val < best_info[1] or (best_max_val == best_info[1] and cust < best_info[4]))):
                        best_info = (regret, best_max_val, best_total_val, cust, best_truck, best_pos)
                regret, best_max_val, best_total_val, cust, best_truck, best_pos = best_info
                routes_repair[best_truck] = routes_repair[best_truck][:best_pos] + [cust] + routes_repair[best_truck][best_pos:]
                dists_repair[best_truck] += insertion_delta(routes_repair[best_truck], best_pos, cust)
                if dists_repair[best_truck] > current_max_repair:
                    current_max_repair = dists_repair[best_truck]
                unassigned.remove(cust)
            new_routes_final = routes_repair
            new_dists_final = dists_repair
        else:  # greedy-max: insert to minimize max distance directly (using best_new_max only)
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(removed)
            current_max_repair = max(dists_repair)
            for cust in unassigned:
                best_truck = None
                best_pos = None
                best_new_max = float('inf')
                for t, route in enumerate(routes_repair):
                    old_dist = dists_repair[t]
                    for pos in range(1, len(route)):
                        delta = insertion_delta(route, pos, cust)
                        new_dist = old_dist + delta
                        new_max = max(current_max_repair, new_dist)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_truck = t
                            best_pos = pos
                routes_repair[best_truck] = routes_repair[best_truck][:best_pos] + [cust] + routes_repair[best_truck][best_pos:]
                dists_repair[best_truck] += insertion_delta(routes_repair[best_truck], best_pos, cust)
                if dists_repair[best_truck] > current_max_repair:
                    current_max_repair = dists_repair[best_truck]
            new_routes_final = routes_repair
            new_dists_final = dists_repair

        new_max = max(new_dists_final)
        new_total = sum(new_dists_final)
        accepted = False
        # RRT acceptance with narrowed beta
        threshold = best_max * (1.0 + beta)
        if new_max <= threshold:
            accepted = True
            current_routes = [list(r) for r in new_routes_final]
            current_dists = list(new_dists_final)
            current_max = new_max
            current_total = new_total
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes_final]
                best_dists = list(new_dists_final)
                # Apply 2-opt to each route of the new best solution
                for t in range(truck_count):
                    best_routes[t] = two_opt_route(best_routes[t])
                best_dists = [route_distance(r) for r in best_routes]
                best_max = max(best_dists)
                best_total = sum(best_dists)
                report_best_vrp(best_routes)
                destroy_scores[destroy_op] += score_best
                repair_scores[repair_op] += score_best
                no_improve_iter = 0
                last_best_iter = it
            else:
                destroy_scores[destroy_op] += score_accepted
                repair_scores[repair_op] += score_accepted
                no_improve_iter += 1
        else:
            destroy_scores[destroy_op] += score_rejected
            repair_scores[repair_op] += score_rejected
            no_improve_iter += 1

        # If no improvement for 50 iterations, intensify current solution via 2-opt
        if no_improve_iter >= 50:
            no_improve_iter = 0
            for t in range(truck_count):
                current_routes[t] = two_opt_route(current_routes[t])
            current_dists = [route_distance(r) for r in current_routes]
            current_max = max(current_dists)
            current_total = sum(current_dists)

    return best_routes
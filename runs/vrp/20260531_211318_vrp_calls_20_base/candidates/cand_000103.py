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

    # Initial construction (same as parent)
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
                delta = insertion_delta(route, pos, cust)
                new_dist = route_dists[t] + delta
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists) + delta
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

    # Parameters adjusted for exploitation: smaller removal, deterministic acceptance
    max_iter = min(2000, 15 * n)
    removal_fraction = 0.15  # smaller to focus
    num_removals = max(1, int(removal_fraction * (n - 1)))

    # Only worst removal (deterministic for exploitation) and greedy repair (deterministic)
    destroy_scores = [1.0]  # only worst removal
    repair_scores = [1.0]   # only greedy repair
    score_best = 3.0
    score_accepted = 1.0
    score_rejected = 0.0

    no_improve_iter = 0
    restart_threshold = int(0.15 * max_iter)  # more sensitive

    for it in range(max_iter):
        # Destroy: worst removal always
        all_contribs = []
        for t, route in enumerate(current_routes):
            if len(route) <= 2:
                continue
            for pos in range(1, len(route)-1):
                contrib = removal_delta(route, pos)
                all_contribs.append((contrib, t, pos, route[pos]))
        all_contribs.sort(key=lambda x: x[0], reverse=True)
        to_remove = set()
        for contrib, t, pos, cust in all_contribs[:num_removals]:
            to_remove.add(cust)
        new_routes = []
        new_dists = []
        for t, route in enumerate(current_routes):
            new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
            new_routes.append(new_route)
            new_dists.append(route_distance(new_route))
        removed = list(to_remove)

        # Repair: greedy repair (deterministic)
        routes_repair = [list(r) for r in new_routes]
        dists_repair = list(new_dists)
        unassigned = list(removed)
        current_max_repair = max(dists_repair)
        for cust in unassigned:
            best_truck = None
            best_pos = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            best_delta = None
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
                        best_delta = delta
            route = routes_repair[best_truck]
            routes_repair[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            dists_repair[best_truck] += best_delta
            if dists_repair[best_truck] > current_max_repair:
                current_max_repair = dists_repair[best_truck]
        new_routes_final = routes_repair
        new_dists_final = dists_repair

        # Deterministic acceptance: only if improves max or same max but better total
        new_max = max(new_dists_final)
        new_total = sum(new_dists_final)
        improved = False
        if new_max < current_max - 1e-9:
            improved = True
        elif abs(new_max - current_max) < 1e-9 and new_total < current_total - 1e-9:
            improved = True

        if improved:
            current_routes = [list(r) for r in new_routes_final]
            current_dists = list(new_dists_final)
            current_max = new_max
            current_total = new_total
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total - 1e-9):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes_final]
                best_dists = list(new_dists_final)
                report_best_vrp(best_routes)
                no_improve_iter = 0
                destroy_scores[0] += score_best
                repair_scores[0] += score_best
            else:
                no_improve_iter += 1
                destroy_scores[0] += score_accepted
                repair_scores[0] += score_accepted
        else:
            no_improve_iter += 1
            destroy_scores[0] += score_rejected
            repair_scores[0] += score_rejected

        # Stagnation restart from best with small perturbation
        if no_improve_iter >= restart_threshold:
            # Perturb best solution by removing a small fraction and reinserting greedily
            perturb_fraction = 0.1
            num_perturb = max(1, int(perturb_fraction * (n - 1)))
            all_customers = [c for r in best_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:num_perturb])
            new_routes = []
            new_dists = []
            for route in best_routes:
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(to_remove)
            current_max_repair = max(dists_repair)
            for cust in unassigned:
                best_truck = None
                best_pos = None
                best_new_max = float('inf')
                best_new_total = float('inf')
                best_delta = None
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
                            best_delta = delta
                route = routes_repair[best_truck]
                routes_repair[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
                dists_repair[best_truck] += best_delta
                if dists_repair[best_truck] > current_max_repair:
                    current_max_repair = dists_repair[best_truck]
            current_routes = routes_repair
            current_dists = dists_repair
            current_max = max(current_dists)
            current_total = sum(current_dists)
            no_improve_iter = 0

    # Intensified local search on best solution: multiple passes of 2-opt and Or-opt
    max_opt_iter = 200  # increased from 100
    for _ in range(max_opt_iter):
        improved = False
        for t, route in enumerate(best_routes):
            if len(route) <= 3:
                continue
            # 2-opt
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dists[t] - 1e-9:
                        new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                        new_total = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total - 1e-9):
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
            # Or-opt: relocate a segment of length 1,2,3
            if len(route) > 4:
                for length in range(1, 4):
                    for start in range(1, len(route) - length):
                        segment = route[start:start+length]
                        rest = route[:start] + route[start+length:]
                        for insert_pos in range(1, len(rest)):
                            if insert_pos == start:
                                continue
                            new_route = rest[:insert_pos] + segment + rest[insert_pos:]
                            new_dist = route_distance(new_route)
                            if new_dist < best_dists[t] - 1e-9:
                                new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                                new_total = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
                                if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total - 1e-9):
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
            if improved:
                break
        if not improved:
            break

    return best_routes
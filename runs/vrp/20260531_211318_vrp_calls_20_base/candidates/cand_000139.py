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

    # Adaptive scores: 3 destroy (worst, random, shaw), 2 repair (greedy, regret-3)
    destroy_scores = [2.0, 1.0, 1.5]  # favor worst and shaw
    repair_scores = [2.0, 2.0]        # greedy and regret-3, no random
    score_best = 5.0
    score_accepted = 1.0
    score_rejected = 0.0

    # Parameters: more conservative
    max_iter = min(3000, 20 * n)
    removal_start = 0.2
    removal_end = 0.05
    beta_start = 0.01
    beta_end = 0.0
    no_improve_iter = 0
    restart_threshold = int(0.2 * max_iter)  # less frequent restart
    large_removal_fraction = 0.5

    for it in range(max_iter):
        removal_fraction = removal_start + (removal_end - removal_start) * (it / max_iter)
        num_removals = max(1, int(removal_fraction * (n - 1)))
        beta = beta_start + (beta_end - beta_start) * (it / max_iter)

        # Select destroy operator
        total_d = sum(destroy_scores)
        destroy_probs = [s / total_d for s in destroy_scores]
        destroy_op = random.choices([0, 1, 2], weights=destroy_probs)[0]

        # Destroy phase
        if destroy_op == 0:  # worst removal
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
        elif destroy_op == 1:  # random removal
            all_customers = [c for r in current_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:num_removals])
        else:  # Shaw removal
            all_cust = [c for r in current_routes for c in r[1:-1]]
            if len(all_cust) == 0:
                to_remove = set()
            else:
                seed = random.choice(all_cust)
                dist_to_seed = [(dist[seed][c], c) for c in all_cust if c != seed]
                dist_to_seed.sort(key=lambda x: x[0])
                to_remove = {seed}
                for _, c in dist_to_seed[:num_removals-1]:
                    to_remove.add(c)

        new_routes = []
        new_dists = []
        for t, route in enumerate(current_routes):
            new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
            new_routes.append(new_route)
            new_dists.append(route_distance(new_route))
        removed = list(to_remove)

        # Select repair operator (only greedy or regret-3)
        total_r = sum(repair_scores)
        repair_probs = [s / total_r for s in repair_scores]
        repair_op = random.choices([0, 1], weights=repair_probs)[0]

        # Repair phase
        if repair_op == 0:  # greedy
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
        else:  # regret-3 repair
            routes_repair = [list(r) for r in new_routes]
            dists_repair = list(new_dists)
            unassigned = list(removed)
            current_max_repair = max(dists_repair)
            while unassigned:
                best_regret = -1
                best_cust = None
                best_insert = None
                for cust in unassigned:
                    candidates = []
                    for t, route in enumerate(routes_repair):
                        old_dist = dists_repair[t]
                        for pos in range(1, len(route)):
                            delta = insertion_delta(route, pos, cust)
                            new_dist = old_dist + delta
                            new_max = max(current_max_repair, new_dist)
                            new_total = sum(dists_repair) + delta
                            candidates.append((new_max, new_total, delta, t, pos))
                    candidates.sort(key=lambda x: (x[0], x[1]))
                    if len(candidates) >= 3:
                        regret = (candidates[1][0] - candidates[0][0]) + (candidates[2][0] - candidates[0][0])
                    elif len(candidates) == 2:
                        regret = candidates[1][0] - candidates[0][0]
                    else:
                        regret = float('inf')
                    if regret > best_regret or (regret == best_regret and (candidates[0][0] < best_insert[0] or (candidates[0][0] == best_insert[0] and candidates[0][1] < best_insert[1]))):
                        best_regret = regret
                        best_cust = cust
                        best_insert = candidates[0]
                t = best_insert[3]
                pos = best_insert[4]
                delta = best_insert[2]
                route = routes_repair[t]
                routes_repair[t] = route[:pos] + [best_cust] + route[pos:]
                dists_repair[t] += delta
                if dists_repair[t] > current_max_repair:
                    current_max_repair = dists_repair[t]
                unassigned.remove(best_cust)
            new_routes_final = routes_repair
            new_dists_final = dists_repair

        new_max = max(new_dists_final)
        new_total = sum(new_dists_final)
        accepted = False
        # Very tight acceptance: only accept if new_max <= best_max * (1+beta)
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
                report_best_vrp(best_routes)
                no_improve_iter = 0
                destroy_scores[destroy_op] += score_best
                repair_scores[repair_op] += score_best
            else:
                no_improve_iter += 1
                destroy_scores[destroy_op] += score_accepted
                repair_scores[repair_op] += score_accepted
        else:
            no_improve_iter += 1
            destroy_scores[destroy_op] += score_rejected
            repair_scores[repair_op] += score_rejected

        # Restart if stuck
        if no_improve_iter >= restart_threshold:
            large_removal_count = max(1, int(large_removal_fraction * (n - 1)))
            all_customers = [c for r in current_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:large_removal_count])
            new_routes = []
            new_dists = []
            for route in current_routes:
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
            # greedy repair
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
            # No reset of beta, keep it small

        # Periodic 2-opt on best solution (every 50 iterations)
        if it % 50 == 0:
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

    # Final post-optimization: 2-opt on best solution
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
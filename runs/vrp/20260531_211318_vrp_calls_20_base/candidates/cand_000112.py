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

    # Adaptive scores with exponential decay
    destroy_scores = [1.0, 1.0]
    repair_scores = [1.0, 1.0]
    decay = 0.9
    score_best = 3.0
    score_accepted = 1.0
    score_rejected = 0.0

    # Parameters
    max_iter = min(3000, 20 * n)
    removal_fraction = 0.2  # fixed for simplicity, can still be dynamic
    # RRT deviation allowance: start high, decrease to 0
    dev_start = best_max * 0.2
    dev_end = 0.0
    # Stagnation
    no_improve_iter = 0
    restart_threshold = int(0.2 * max_iter)
    # For local search period
    ls_period = 100

    for it in range(max_iter):
        num_removals = max(1, int(removal_fraction * (n - 1)))

        # Select destroy
        total_d = sum(destroy_scores)
        destroy_probs = [s / total_d for s in destroy_scores]
        destroy_op = random.choices([0, 1], weights=destroy_probs)[0]

        # Destroy
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
            new_routes = []
            new_dists = []
            for t, route in enumerate(current_routes):
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
        else:  # random removal
            all_customers = [c for r in current_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:num_removals])
            new_routes = []
            new_dists = []
            for route in current_routes:
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
        removed = list(to_remove)

        # Select repair
        total_r = sum(repair_scores)
        repair_probs = [s / total_r for s in repair_scores]
        repair_op = random.choices([0, 1], weights=repair_probs)[0]

        # Repair
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
        else:  # regret-2
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
                    best_delta = None
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
                                best_delta = delta
                            elif new_max < second_best_max or (new_max == second_best_max and new_total < second_best_total):
                                second_best_max = new_max
                                second_best_total = new_total
                    regret = (second_best_max - best_max_val) if second_best_max != float('inf') else float('inf')
                    if best_info is None or regret > best_info[0] or (regret == best_info[0] and (best_max_val < best_info[1] or (best_max_val == best_info[1] and cust < best_info[4]))):
                        best_info = (regret, best_max_val, best_total_val, cust, best_truck, best_pos, best_delta)
                regret, best_max_val, best_total_val, cust, best_truck, best_pos, best_delta = best_info
                route = routes_repair[best_truck]
                routes_repair[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
                dists_repair[best_truck] += best_delta
                if dists_repair[best_truck] > current_max_repair:
                    current_max_repair = dists_repair[best_truck]
                unassigned.remove(cust)
            new_routes_final = routes_repair
            new_dists_final = dists_repair

        new_max = max(new_dists_final)
        new_total = sum(new_dists_final)
        # Record-to-Record Travel acceptance
        dev = dev_start + (dev_end - dev_start) * (it / max_iter)
        if new_max <= best_max + dev:
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

        # Decay scores
        destroy_scores = [s * decay for s in destroy_scores]
        repair_scores = [s * decay for s in repair_scores]

        # Periodic 2-opt on best solution
        if it % ls_period == 0 and it > 0:
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
                                new_max2 = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                                new_total2 = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
                                if new_max2 < best_max - 1e-9 or (abs(new_max2 - best_max) < 1e-9 and new_total2 < best_total):
                                    best_routes[t] = new_route
                                    best_dists[t] = new_dist
                                    best_max = new_max2
                                    best_total = new_total2
                                    report_best_vrp(best_routes)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

        # Stagnation: shake best solution
        if no_improve_iter >= restart_threshold:
            # Find longest route
            max_dist_idx = max(range(truck_count), key=lambda t: best_dists[t])
            longest_route = best_routes[max_dist_idx]
            if len(longest_route) > 2:
                # Remove a customer from longest route (not first/last)
                pos = random.randint(1, len(longest_route)-2)
                cust = longest_route[pos]
                new_longest = longest_route[:pos] + longest_route[pos+1:]
                best_routes[max_dist_idx] = new_longest
                best_dists[max_dist_idx] = route_distance(new_longest)
                # Reinsert greedily into any route
                unassigned = [cust]
                routes_temp = [list(r) for r in best_routes]
                dists_temp = list(best_dists)
                current_max_temp = max(dists_temp)
                # Greedy insertion
                for cust2 in unassigned:
                    best_truck = None
                    best_pos = None
                    best_new_max = float('inf')
                    best_new_total = float('inf')
                    best_delta = None
                    for t, route in enumerate(routes_temp):
                        old_dist = dists_temp[t]
                        for pos2 in range(1, len(route)):
                            delta = insertion_delta(route, pos2, cust2)
                            new_dist = old_dist + delta
                            new_max = max(current_max_temp, new_dist)
                            new_total = sum(dists_temp) + delta
                            if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                                best_new_max = new_max
                                best_new_total = new_total
                                best_truck = t
                                best_pos = pos2
                                best_delta = delta
                    route = routes_temp[best_truck]
                    routes_temp[best_truck] = route[:best_pos] + [cust2] + route[best_pos:]
                    dists_temp[best_truck] += best_delta
                    if dists_temp[best_truck] > current_max_temp:
                        current_max_temp = dists_temp[best_truck]
                # Update best if improved
                new_max = max(dists_temp)
                new_total = sum(dists_temp)
                if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                    best_routes = routes_temp
                    best_dists = dists_temp
                    best_max = new_max
                    best_total = new_total
                    report_best_vrp(best_routes)
                # Reset current solution to best
                current_routes = [list(r) for r in best_routes]
                current_dists = list(best_dists)
                current_max = best_max
                current_total = best_total
            no_improve_iter = 0

    # Final 2-opt on best solution
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
                        new_max2 = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                        new_total2 = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
                        if new_max2 < best_max - 1e-9 or (abs(new_max2 - best_max) < 1e-9 and new_total2 < best_total):
                            best_routes[t] = new_route
                            best_dists[t] = new_dist
                            best_max = new_max2
                            best_total = new_total2
                            report_best_vrp(best_routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    return best_routes
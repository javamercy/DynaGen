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
        best_new_max = float('inf')
        best_new_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_dist = route_dists[t] + insertion_delta(route, pos, cust)
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists) + insertion_delta(route, pos, cust)
                if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                    best_new_max = new_max
                    best_new_total = new_total
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

    # Main loop: simple destroy (worst) and repair (greedy)
    max_iter = min(2000, 20 * n)
    removal_fraction = 0.2
    for it in range(max_iter):
        num_removals = max(1, int(removal_fraction * (n - 1)))
        # Worst removal
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
        removed = list(to_remove)

        # Greedy insertion
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
            route = routes_repair[best_truck]
            routes_repair[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            dists_repair[best_truck] += insertion_delta(route, best_pos, cust)
            if dists_repair[best_truck] > current_max_repair:
                current_max_repair = dists_repair[best_truck]
        new_routes_final = routes_repair
        new_dists_final = dists_repair

        new_max = max(new_dists_final)
        new_total = sum(new_dists_final)
        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
            best_max = new_max
            best_total = new_total
            best_routes = [list(r) for r in new_routes_final]
            best_dists = list(new_dists_final)
            report_best_vrp(best_routes)
            current_routes = [list(r) for r in new_routes_final]
            current_dists = list(new_dists_final)
            current_max = new_max
            current_total = new_total
        elif new_max <= current_max + 1e-9:
            current_routes = [list(r) for r in new_routes_final]
            current_dists = list(new_dists_final)
            current_max = new_max
            current_total = new_total

    # Intensification: bottleneck local search on best solution
    max_opt_iter = 100
    for _ in range(max_opt_iter):
        improved = False
        max_val = max(best_dists)
        candidate_routes = [t for t, d in enumerate(best_dists) if d == max_val]
        for t in candidate_routes:
            route = best_routes[t]
            # 2-opt within route
            if len(route) > 3:
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
            # Relocate customers from this route to other routes
            if len(route) <= 2:
                continue
            for pos in range(1, len(route) - 1):
                cust = route[pos]
                temp_route = route[:pos] + route[pos+1:]
                temp_dist = route_distance(temp_route)
                best_move = None
                best_new_max = float('inf')
                best_new_total = float('inf')
                for t2 in range(truck_count):
                    if t2 == t:
                        r2 = temp_route
                    else:
                        r2 = best_routes[t2]
                    old_dist2 = best_dists[t2] if t2 != t else temp_dist
                    for p in range(1, len(r2)):
                        delta = insertion_delta(r2, p, cust)
                        new_dist2 = old_dist2 + delta
                        new_dists_temp = [d for i, d in enumerate(best_dists) if i != t and i != t2]
                        if t != t2:
                            new_max_temp = max(new_dists_temp + [temp_dist, new_dist2])
                            new_total_temp = sum(new_dists_temp) + temp_dist + new_dist2
                        else:
                            new_max_temp = max(new_dists_temp + [new_dist2])
                            new_total_temp = sum(new_dists_temp) + new_dist2
                        if new_max_temp < best_new_max or (new_max_temp == best_new_max and new_total_temp < best_new_total):
                            best_new_max = new_max_temp
                            best_new_total = new_total_temp
                            best_move = (t2, p, delta, temp_route, temp_dist)
                if best_move and best_new_max < best_max - 1e-9:
                    t2, p, delta, temp_route, temp_dist = best_move
                    if t2 == t:
                        best_routes[t] = temp_route[:p] + [cust] + temp_route[p:]
                        best_dists[t] = temp_dist + delta
                    else:
                        best_routes[t] = temp_route
                        best_dists[t] = temp_dist
                        best_routes[t2] = best_routes[t2][:p] + [cust] + best_routes[t2][p:]
                        best_dists[t2] += delta
                    best_max = best_new_max
                    best_total = best_new_total
                    report_best_vrp(best_routes)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Cross-route 2-opt* moves on all route pairs
    max_opt2_iter = 50
    for _ in range(max_opt2_iter):
        improved = False
        for t1 in range(truck_count):
            for t2 in range(t1 + 1, truck_count):
                r1 = best_routes[t1]
                r2 = best_routes[t2]
                if len(r1) <= 2 and len(r2) <= 2:
                    continue
                for i in range(1, len(r1) - 1):
                    for j in range(1, len(r2) - 1):
                        new_r1 = r1[:i] + r2[j:]
                        new_r2 = r2[:j] + r1[i:]
                        d1 = route_distance(new_r1)
                        d2 = route_distance(new_r2)
                        new_dists_temp = [d for k, d in enumerate(best_dists) if k != t1 and k != t2]
                        new_max = max(new_dists_temp + [d1, d2])
                        new_total = sum(new_dists_temp) + d1 + d2
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                            best_routes[t1] = new_r1
                            best_routes[t2] = new_r2
                            best_dists[t1] = d1
                            best_dists[t2] = d2
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
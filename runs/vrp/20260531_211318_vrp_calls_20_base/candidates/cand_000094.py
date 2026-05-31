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

    # Initial construction: greedy insertion minimizing max distance
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

    max_iter = min(2000, 20 * n)
    removal_fraction = 0.15
    num_removals = max(1, int(removal_fraction * (n - 1)))
    no_improve_iter = 0
    restart_threshold = int(0.15 * max_iter)

    for it in range(max_iter):
        # Destroy: worst removal
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

        # Repair: greedy insertion minimizing max distance, tie-breaking total distance
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

        new_max = max(dists_repair)
        new_total = sum(dists_repair)
        accepted = False
        # Only accept if improves (strict hill-climbing)
        if new_max < current_max - 1e-9 or (abs(new_max - current_max) < 1e-9 and new_total < current_total):
            accepted = True
            current_routes = [list(r) for r in routes_repair]
            current_dists = list(dists_repair)
            current_max = new_max
            current_total = new_total
            # Local search on each route: 2-opt and relocate
            improved_local = True
            while improved_local:
                improved_local = False
                for t in range(truck_count):
                    route = current_routes[t]
                    if len(route) <= 3:
                        continue
                    # 2-opt
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_dist = route_distance(new_route)
                            if new_dist < current_dists[t] - 1e-9:
                                current_dists[t] = new_dist
                                current_routes[t] = new_route
                                improved_local = True
                                break
                        if improved_local:
                            break
                    if improved_local:
                        break
                # Relocate: try moving a customer to a better position within same route (Or-opt: move one node)
                for t in range(truck_count):
                    route = current_routes[t]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-1):
                        cust = route[i]
                        for j in range(1, len(route)-1):
                            if j == i or j == i-1 or j == i+1:
                                continue
                            new_route = route[:i] + route[i+1:]
                            new_route = new_route[:j] + [cust] + new_route[j:]
                            new_dist = route_distance(new_route)
                            if new_dist < current_dists[t] - 1e-9:
                                current_dists[t] = new_dist
                                current_routes[t] = new_route
                                improved_local = True
                                break
                        if improved_local:
                            break
                    if improved_local:
                        break
            # Update current_max and current_total after local search
            current_max = max(current_dists)
            current_total = sum(current_dists)
            # Check best
            if current_max < best_max - 1e-9 or (abs(current_max - best_max) < 1e-9 and current_total < best_total):
                best_max = current_max
                best_total = current_total
                best_routes = [list(r) for r in current_routes]
                best_dists = list(current_dists)
                report_best_vrp(best_routes)
                no_improve_iter = 0
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1

        # Restart if stuck
        if no_improve_iter >= restart_threshold:
            large_removal_count = max(1, int(0.4 * (n - 1)))
            all_customers = [c for r in current_routes for c in r[1:-1]]
            random.shuffle(all_customers)
            to_remove = set(all_customers[:large_removal_count])
            new_routes = []
            new_dists = []
            for route in current_routes:
                new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
                new_routes.append(new_route)
                new_dists.append(route_distance(new_route))
            # Greedy repair
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

    # Post-optimization: 2-opt on best solution
    improved = True
    while improved:
        improved = False
        for t in range(truck_count):
            route = best_routes[t]
            if len(route) <= 3:
                continue
            # 2-opt
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dists[t] - 1e-9:
                        best_dists[t] = new_dist
                        best_routes[t] = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        # Also try inter-route customer moves: move a customer from longest route to a shorter one
        if not improved:
            # Find route with max distance
            max_idx = max(range(truck_count), key=lambda i: best_dists[i])
            for cust_pos in range(1, len(best_routes[max_idx])-1):
                cust = best_routes[max_idx][cust_pos]
                for other_t in range(truck_count):
                    if other_t == max_idx:
                        continue
                    for insert_pos in range(1, len(best_routes[other_t])):
                        new_route_max = best_routes[max_idx][:cust_pos] + best_routes[max_idx][cust_pos+1:]
                        new_route_other = best_routes[other_t][:insert_pos] + [cust] + best_routes[other_t][insert_pos:]
                        new_dist_max = route_distance(new_route_max)
                        new_dist_other = route_distance(new_route_other)
                        new_max = max(best_dists[:max_idx] + [new_dist_max] + best_dists[max_idx+1:other_t] + [new_dist_other] + best_dists[other_t+1:])
                        new_total = sum(best_dists) - best_dists[max_idx] - best_dists[other_t] + new_dist_max + new_dist_other
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                            best_routes[max_idx] = new_route_max
                            best_routes[other_t] = new_route_other
                            best_dists[max_idx] = new_dist_max
                            best_dists[other_t] = new_dist_other
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
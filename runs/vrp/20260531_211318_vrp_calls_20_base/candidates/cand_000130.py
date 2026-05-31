import numpy as np
import random
import math

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

    # Parameters
    max_iter = min(2000, 20 * n)
    removal_start = 0.3
    removal_end = 0.05
    beta_start = 0.05
    beta_end = 0.005

    # Local search setting: apply 2-opt on best solution every 10 iterations
    ls_interval = 10

    for it in range(max_iter):
        removal_fraction = removal_start + (removal_end - removal_start) * (it / max_iter)
        num_removals = max(1, int(removal_fraction * (n - 1)))
        beta = beta_start + (beta_end - beta_start) * (it / max_iter)

        # Biased removal: select route with probability proportional to route distance
        total_current_dist = sum(current_dists)
        if total_current_dist == 0:
            route_weights = [1.0 / truck_count] * truck_count
        else:
            route_weights = [d / total_current_dist for d in current_dists]
        # Select one route to remove from (weighted by route distance)
        # Actually, we want to remove customers from the worst routes more often.
        # We'll select a route according to weights, then randomly remove customers from that route.
        # But we need to remove from all routes to get num_removals customers? 
        # Approach: select a route with probability = weight, then remove a random customer from that route, repeat until we have enough.
        to_remove = set()
        customers_pool = [(t, i) for t in range(truck_count) for i in range(1, len(current_routes[t])-1)]
        if not customers_pool:
            continue
        # Actually, we want biased selection of route, then random customer from it.
        # Since we need multiple removals, we can do weighted route selection each time.
        # To avoid removing too many from one route, we can do a single route selection and then remove random customers from it up to num_removals.
        # Simpler: select a route with probability proportional to its distance, then remove all its customers? But that may remove too many.
        # We'll select a route with probability proportional to route distance, then remove a random customer from that route.
        # Repeat until we have enough removals.
        removed_from_route = {t: 0 for t in range(truck_count)}
        while len(to_remove) < num_removals and len(customers_pool) > 0:
            # Weighted choice of route
            if total_current_dist == 0:
                t = random.randrange(truck_count)
            else:
                r = random.random() * total_current_dist
                cumulative = 0.0
                t = 0
                for ti in range(truck_count):
                    cumulative += current_dists[ti]
                    if r <= cumulative:
                        t = ti
                        break
            # Check if route has customers left to remove
            route = current_routes[t]
            if len(route) <= 2:
                continue
            # Remove a random customer from that route
            pos = random.randint(1, len(route)-2)
            cust = route[pos]
            if cust not in to_remove:
                to_remove.add(cust)
                removed_from_route[t] += 1
        # Actually, simple: just select random customers from all customers but give higher weight to customers in longer routes? 
        # For simplicity, we will do random removal (like parent) but after removal we will focus local search. 
        # But the instruction says improve exploitation. Let's keep removal simple to avoid complexity.
        # Revert to random removal as parent: simpler and robust.
        # We'll keep random removal.
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

        # Regret-2 repair
        routes_repair = [list(r) for r in new_routes]
        dists_repair = list(new_dists)
        unassigned = list(removed)
        current_max_repair = max(dists_repair)
        while unassigned:
            # For each unassigned customer, compute best and second best insertion cost
            best_info = []  # (regret, truck, pos, delta)
            for cust in unassigned:
                best_delta = float('inf')
                second_best_delta = float('inf')
                best_truck = None
                best_pos = None
                for t, route in enumerate(routes_repair):
                    for pos in range(1, len(route)):
                        delta = insertion_delta(route, pos, cust)
                        if delta < best_delta:
                            second_best_delta = best_delta
                            best_delta = delta
                            best_truck = t
                            best_pos = pos
                        elif delta < second_best_delta:
                            second_best_delta = delta
                regret = second_best_delta - best_delta
                best_info.append((regret, cust, best_truck, best_pos, best_delta))
            # Choose customer with highest regret (ties broken by smallest best_delta then smallest cust)
            best_info.sort(key=lambda x: (-x[0], x[4], x[1]))
            _, cust, t, pos, delta = best_info[0]
            route = routes_repair[t]
            routes_repair[t] = route[:pos] + [cust] + route[pos:]
            dists_repair[t] += delta
            if dists_repair[t] > current_max_repair:
                current_max_repair = dists_repair[t]
            unassigned.remove(cust)
        new_max = max(dists_repair)
        new_total = sum(dists_repair)

        # RRT acceptance
        threshold = best_max * (1.0 + beta)
        if new_max <= threshold:
            current_routes = [list(r) for r in routes_repair]
            current_dists = list(dists_repair)
            current_max = new_max
            current_total = new_total
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in routes_repair]
                best_dists = list(dists_repair)
                report_best_vrp(best_routes)

        # Local search on best solution every ls_interval iterations
        if it % ls_interval == 0:
            # Apply 2-opt on each route of best_routes
            improved = True
            max_ls_iter = 100
            ls_iter = 0
            while improved and ls_iter < max_ls_iter:
                improved = False
                for t in range(truck_count):
                    route = best_routes[t]
                    if len(route) <= 3:
                        continue
                    best_len = route_distance(route)
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            # 2-opt swap: reverse segment from i to j
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_len = route_distance(new_route)
                            if new_len < best_len - 1e-9:
                                # Update best_routes and best_dists
                                best_routes[t] = new_route
                                best_dists[t] = new_len
                                improved = True
                                # Recompute best_max and best_total
                                new_best_max = max(best_dists)
                                new_best_total = sum(best_dists)
                                # Only accept if best_max does not increase (should not, as we are improving one route)
                                # Actually, if this route was not the max, max remains. So it's safe.
                                # But we need to update best_max and best_total
                                if new_best_max < best_max - 1e-9 or (abs(new_best_max - best_max) < 1e-9 and new_best_total < best_total):
                                    best_max = new_best_max
                                    best_total = new_best_total
                                    report_best_vrp(best_routes)
                                break  # break inner loops after improvement to restart
                        if improved:
                            break
                ls_iter += 1

    # Final local search on best solution
    improved = True
    max_ls_iter = 100
    ls_iter = 0
    while improved and ls_iter < max_ls_iter:
        improved = False
        for t in range(truck_count):
            route = best_routes[t]
            if len(route) <= 3:
                continue
            best_len = route_distance(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = route_distance(new_route)
                    if new_len < best_len - 1e-9:
                        best_routes[t] = new_route
                        best_dists[t] = new_len
                        improved = True
                        new_best_max = max(best_dists)
                        new_best_total = sum(best_dists)
                        if new_best_max < best_max - 1e-9 or (abs(new_best_max - best_max) < 1e-9 and new_best_total < best_total):
                            best_max = new_best_max
                            best_total = new_best_total
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break
        ls_iter += 1

    return best_routes
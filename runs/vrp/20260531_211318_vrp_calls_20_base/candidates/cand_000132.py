import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0,0] for _ in range(truck_count)]
    random.seed(0)  # deterministic seed for reproducibility
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        return sum(dist[route[i], route[i+1]] for i in range(len(route)-1))

    # ---- Nearest neighbor construction (different from farthest-first) ----
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unvisited = list(range(1, n))
    while unvisited:
        # Pick a random unvisited customer to start? Use deterministic: smallest index
        cust = unvisited[0]
        best_truck = None
        best_pos = None
        best_delta = float('inf')
        for t, route in enumerate(routes):
            # Insert at best position for this truck (greedy insertion to minimize increase in route distance)
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                delta = dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]
                if delta < best_delta or (delta == best_delta and (best_truck is None or cust < routes[best_truck][best_pos] if best_pos else True)):
                    best_delta = delta
                    best_truck = t
                    best_pos = pos
        route = routes[best_truck]
        routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
        route_dists[best_truck] += best_delta
        unvisited.remove(cust)

    current_routes = [list(r) for r in routes]
    current_dists = list(route_dists)
    current_max = max(current_dists)
    current_total = sum(current_dists)
    best_routes = [list(r) for r in routes]
    best_dists = list(route_dists)
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    # ---- Simulated annealing parameters ----
    T_start = 0.5 * max(1, best_max)  # scale with problem
    T_end = 1e-3
    max_iter = min(5000, 50 * n)
    T = T_start
    for it in range(max_iter):
        # Cooling: exponential
        T = T_start * math.pow(T_end/T_start, it/(max_iter-1))
        if it == max_iter-1:
            T = T_end

        # Generate neighbor by random move: with 50% relocate, 50% swap
        move_type = random.choice(['relocate', 'swap'])
        new_routes = [list(r) for r in current_routes]
        new_dists = list(current_dists)
        if move_type == 'relocate':
            # choose a random customer from a non-trivial route
            non_trivial = [t for t, r in enumerate(new_routes) if len(r) > 3]
            if not non_trivial:
                continue
            t_from = random.choice(non_trivial)
            route_from = new_routes[t_from]
            # pick random position (excluding depot)
            pos_from = random.randint(1, len(route_from)-2)
            cust = route_from.pop(pos_from)
            # update distance for from-route
            new_dists[t_from] = route_distance(route_from)
            # choose target route and position
            t_to = random.randint(0, truck_count-1)
            route_to = new_routes[t_to]
            # insert at random position (excluding depot)
            pos_to = random.randint(1, len(route_to)-1)
            route_to.insert(pos_to, cust)
            new_dists[t_to] = route_distance(route_to)
        else:  # swap
            # pick two different routes
            t1 = random.randint(0, truck_count-1)
            t2 = random.randint(0, truck_count-1)
            if t1 == t2:
                continue
            r1 = new_routes[t1]
            r2 = new_routes[t2]
            if len(r1) <= 2 or len(r2) <= 2:
                continue
            # pick random customer from each (excluding depots)
            i1 = random.randint(1, len(r1)-2)
            i2 = random.randint(1, len(r2)-2)
            cust1 = r1[i1]
            cust2 = r2[i2]
            # swap
            r1[i1] = cust2
            r2[i2] = cust1
            new_dists[t1] = route_distance(r1)
            new_dists[t2] = route_distance(r2)

        new_max = max(new_dists)
        new_total = sum(new_dists)
        delta = new_max - current_max
        accepted = False
        if delta < 0 or (delta == 0 and new_total < current_total):
            accepted = True
        else:
            # Metropolis acceptance
            if T > 0 and random.random() < math.exp(-delta / T):
                accepted = True
        if accepted:
            current_routes = new_routes
            current_dists = new_dists
            current_max = new_max
            current_total = new_total
            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                best_max = new_max
                best_total = new_total
                best_routes = [list(r) for r in new_routes]
                best_dists = list(new_dists)
                report_best_vrp(best_routes)

    # Post-optimization: simple intra-route 2-opt (limited iterations - kept small for speed)
    for _ in range(100):
        improved = False
        for t, route in enumerate(best_routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dists[t] - 1e-9:
                        new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                        new_total = sum(best_dists) - best_dists[t] + new_dist
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
        if not improved:
            break

    return best_routes
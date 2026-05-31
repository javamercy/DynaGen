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
    current_dists = [route_distance(r) for r in routes]
    current_max = max(current_dists)
    current_total = sum(current_dists)
    best_routes = [list(r) for r in routes]
    best_dists = list(current_dists)
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    # Simulated annealing parameters
    max_iterations = min(5000, 30 * n)
    T = best_max * 0.5  # initial temperature
    alpha = 0.99

    def generate_neighbor(routes_in):
        routes_copy = [list(r) for r in routes_in]
        non_empty = [t for t, r in enumerate(routes_copy) if len(r) > 3]
        if len(non_empty) < 1:
            return None
        move_type = random.randint(0, 2)
        if move_type == 0:  # intra-route swap
            t = random.choice(non_empty)
            route = routes_copy[t]
            customers = route[1:-1]
            if len(customers) < 2:
                return None
            i = random.randint(0, len(customers)-1)
            j = random.randint(0, len(customers)-1)
            while j == i:
                j = random.randint(0, len(customers)-1)
            # swap positions i+1 and j+1
            route[i+1], route[j+1] = route[j+1], route[i+1]
        elif move_type == 1:  # inter-route swap
            # select two different non-empty routes (or same if only one? but then it's intra)
            if len(non_empty) < 2:
                # fallback to intra
                t = random.choice(non_empty)
                route = routes_copy[t]
                customers = route[1:-1]
                if len(customers) < 2:
                    return None
                i = random.randint(0, len(customers)-1)
                j = random.randint(0, len(customers)-1)
                while j == i:
                    j = random.randint(0, len(customers)-1)
                route[i+1], route[j+1] = route[j+1], route[i+1]
            else:
                t1, t2 = random.sample(non_empty, 2)
                route1 = routes_copy[t1]
                route2 = routes_copy[t2]
                cust1 = route1[1:-1]
                cust2 = route2[1:-1]
                if len(cust1) == 0 or len(cust2) == 0:
                    return None
                i = random.randint(0, len(cust1)-1)
                j = random.randint(0, len(cust2)-1)
                # swap customers
                route1[i+1], route2[j+1] = route2[j+1], route1[i+1]
        else:  # inter-route insert (move customer from one route to another)
            # select source and destination routes
            if len(non_empty) < 1:
                return None
            src = random.choice(non_empty)
            # destination can be any route (including empty or same? but moving to same would be intra-insert, which is covered by swap? Keep simple: destination different)
            all_routes = list(range(truck_count))
            if len(all_routes) < 2:
                return None
            dst = random.choice([t for t in all_routes if t != src])
            src_route = routes_copy[src]
            if len(src_route) <= 3:  # only depot
                return None
            # pick random customer from src
            cust_idx = random.randint(1, len(src_route)-2)
            cust = src_route[cust_idx]
            # remove customer from src
            src_route.pop(cust_idx)
            # insert into dst at random position
            dst_route = routes_copy[dst]
            pos = random.randint(1, len(dst_route)-1)
            dst_route.insert(pos, cust)
        return routes_copy

    iteration = 0
    while iteration < max_iterations:
        neighbor = generate_neighbor(current_routes)
        if neighbor is None:
            iteration += 1
            continue
        new_dists = [route_distance(r) for r in neighbor]
        new_max = max(new_dists)
        delta = new_max - current_max
        if delta < 0:
            # accept
            current_routes = [list(r) for r in neighbor]
            current_dists = new_dists
            current_max = new_max
            current_total = sum(new_dists)
            if current_max < best_max - 1e-9 or (abs(current_max - best_max) < 1e-9 and current_total < best_total):
                best_routes = [list(r) for r in neighbor]
                best_dists = new_dists
                best_max = current_max
                best_total = current_total
                report_best_vrp(best_routes)
        else:
            if random.random() < math.exp(-delta / T):
                current_routes = [list(r) for r in neighbor]
                current_dists = new_dists
                current_max = new_max
                current_total = sum(new_dists)
        T *= alpha
        iteration += 1

    # Final 2-opt local search on best solution
    max_opt_iter = 200
    for _ in range(max_opt_iter):
        improved = False
        for t, route in enumerate(best_routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
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
        if not improved:
            break

    return best_routes
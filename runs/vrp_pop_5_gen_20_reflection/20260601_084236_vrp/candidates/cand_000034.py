import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        best_route = route[:]
        best_dist = route_distance(route)
        max_iter = 10  # fixed small number to limit iterations
        for _ in range(max_iter):
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new_route)
                    if d < best_dist - 1e-12:
                        best_dist = d
                        best_route = new_route
                        improved = True
                        break
                if improved:
                    break
            route = best_route
            if not improved:
                break
        return route

    def construct_initial(seed_idx):
        random.seed(seed_idx)
        seeds = random.sample(customers, min(truck_count, len(customers)))
        clusters = [[] for _ in range(truck_count)]
        for i, s in enumerate(seeds):
            clusters[i].append(s)
        remaining = [c for c in customers if c not in seeds]
        for cust in remaining:
            best_dist = float('inf')
            best_cluster = 0
            for i, seed in enumerate(seeds):
                d = distance_matrix[cust, seed]
                if d < best_dist - 1e-12:
                    best_dist = d
                    best_cluster = i
            clusters[best_cluster].append(cust)
        routes = []
        for cluster in clusters:
            if not cluster:
                routes.append([0, 0])
            else:
                unvisited = set(cluster)
                current = 0
                tour = [0]
                while unvisited:
                    next_cust = min(unvisited, key=lambda c: distance_matrix[current, c])
                    tour.append(next_cust)
                    unvisited.remove(next_cust)
                    current = next_cust
                tour.append(0)
                route = two_opt(tour)
                routes.append(route)
        return routes

    best_routes = None
    best_max = float('inf')
    max_restarts = min(5, truck_count)  # reduced restarts
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        current_max = max_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Intra-route 2-opt on each route
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        current_max = max_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Inter-route 2-opt*: single best improvement
        best_improv = None
        best_improv_new_max = float('inf')
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                r1 = routes[t1]
                r2 = routes[t2]
                if len(r1) <= 2 or len(r2) <= 2:
                    continue
                for i in range(1, len(r1)-1):
                    for j in range(1, len(r2)-1):
                        new_r1 = r1[:i+1] + r2[j+1:]
                        new_r2 = r2[:j+1] + r1[i+1:]
                        d1 = route_distance(new_r1)
                        d2 = route_distance(new_r2)
                        other_max = 0.0
                        for idx, r in enumerate(routes):
                            if idx not in (t1, t2):
                                d = route_distance(r)
                                if d > other_max:
                                    other_max = d
                        new_max_candidate = max(d1, d2, other_max)
                        if new_max_candidate < best_improv_new_max - 1e-12:
                            best_improv_new_max = new_max_candidate
                            best_improv = (t1, t2, i, j, new_r1, new_r2)
        if best_improv is not None and best_improv_new_max < best_max - 1e-12:
            t1, t2, i, j, new_r1, new_r2 = best_improv
            routes[t1] = two_opt(new_r1)
            routes[t2] = two_opt(new_r2)
            current_max = max_distance(routes)
            if current_max < best_max - 1e-12:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)

        # Final intra-route 2-opt
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        current_max = max_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    return best_routes
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
        improved = True
        while improved:
            improved = False
            best_route = route[:]
            best_dist = route_distance(route)
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
                if improved:
                    break
            route = best_route
        return route

    def construct_initial(seed_idx):
        random.seed(seed_idx)
        # Choose seeds as farthest customers from depot
        dist_from_depot = [(distance_matrix[0, c], c) for c in customers]
        dist_from_depot.sort(key=lambda x: -x[0])
        seeds = [c for _, c in dist_from_depot[:truck_count]]
        clusters = [[] for _ in range(truck_count)]
        for i, s in enumerate(seeds):
            clusters[i].append(s)
        remaining = [c for c in customers if c not in seeds]
        # Assign each remaining customer to the nearest seed (closest in distance)
        for cust in remaining:
            best_dist = float('inf')
            best_cluster = 0
            for i, seed in enumerate(seeds):
                d = distance_matrix[cust, seed]
                if d < best_dist - 1e-12:
                    best_dist = d
                    best_cluster = i
            clusters[best_cluster].append(cust)
        # Build routes using nearest neighbor, then 2-opt
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

    def local_search(routes, best_routes, best_max):
        improved = False
        # Identify longest route index
        max_dist = max_distance(routes)
        longest_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]))
        # Relocate customers from longest route to others
        route_long = routes[longest_idx]
        if len(route_long) > 3:
            for cust in route_long[1:-1]:
                new_route_long = [0] + [c for c in route_long[1:-1] if c != cust] + [0]
                new_route_long = two_opt(new_route_long)
                for other_idx in range(truck_count):
                    if other_idx == longest_idx:
                        continue
                    route_other = routes[other_idx]
                    for i in range(1, len(route_other)):
                        new_route_other = route_other[:i] + [cust] + route_other[i:]
                        new_route_other = two_opt(new_route_other)
                        new_max = max(route_distance(new_route_long), route_distance(new_route_other),
                                      max(route_distance(r) for idx, r in enumerate(routes) if idx not in [longest_idx, other_idx]))
                        if new_max < best_max - 1e-12:
                            routes[longest_idx] = new_route_long
                            routes[other_idx] = new_route_other
                            best_max = new_max
                            if best_max < max_distance(best_routes):
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            return routes, best_routes, best_max, True
        # Swap between longest and any other route
        for other_idx in range(truck_count):
            if other_idx == longest_idx:
                continue
            route_other = routes[other_idx]
            if len(route_long) <= 3 or len(route_other) <= 2:
                continue
            for cust_a in route_long[1:-1]:
                for cust_b in route_other[1:-1]:
                    new_route_long = [0] + [c for c in route_long[1:-1] if c != cust_a] + [cust_b] + [0]
                    new_route_other = [0] + [c for c in route_other[1:-1] if c != cust_b] + [cust_a] + [0]
                    new_route_long = two_opt(new_route_long)
                    new_route_other = two_opt(new_route_other)
                    new_max = max(route_distance(new_route_long), route_distance(new_route_other),
                                  max(route_distance(r) for idx, r in enumerate(routes) if idx not in [longest_idx, other_idx]))
                    if new_max < best_max - 1e-12:
                        routes[longest_idx] = new_route_long
                        routes[other_idx] = new_route_other
                        best_max = new_max
                        if best_max < max_distance(best_routes):
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                        return routes, best_routes, best_max, True
        return routes, best_routes, best_max, False

    best_routes = None
    best_max = float('inf')
    max_outer = max(truck_count, 5)
    for restart in range(max_outer):
        routes = construct_initial(restart)
        current_max = max_distance(routes)
        if current_max < best_max - 1e-12:
            best_routes = [r[:] for r in routes]
            best_max = current_max
            report_best_vrp(best_routes)
        # Local search without perturbation: multiple passes
        for _ in range(max_outer):
            improved = True
            iters = 0
            max_iters = n * truck_count
            while improved and iters < max_iters:
                routes, best_routes, best_max, improved = local_search(routes, best_routes, best_max)
                iters += 1
            # After local search, re-evaluate
            current_max = max_distance(routes)
            if current_max < best_max - 1e-12:
                best_routes = [r[:] for r in routes]
                best_max = current_max
                report_best_vrp(best_routes)
    return best_routes
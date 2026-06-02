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

    def compute_route_distances(routes):
        return [route_distance(r) for r in routes]

    def improve_by_relocate(routes, dists, best_max, best_routes):
        # find the route with max distance
        max_idx = max(range(len(dists)), key=lambda i: dists[i])
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            return False
        max_custs = max_route[1:-1]
        best_improvement = False
        for cust in max_custs:
            for other_idx in range(len(routes)):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                # try inserting cust into other_route
                best_pos = None
                best_other_dist = float('inf')
                for pos in range(1, len(other_route)):
                    new_route_b = other_route[:pos] + [cust] + other_route[pos:]
                    d = route_distance(new_route_b)
                    if d < best_other_dist - 1e-12:
                        best_other_dist = d
                        best_pos = pos
                if best_pos is None:
                    continue
                new_route_a = [0] + [c for c in max_route[1:-1] if c != cust] + [0]
                new_route_a = two_opt(new_route_a)
                new_route_b = two_opt(other_route[:best_pos] + [cust] + other_route[best_pos:])
                new_dist_a = route_distance(new_route_a)
                new_dist_b = route_distance(new_route_b)
                old_max = max(dists)
                new_max = max(new_dist_a, new_dist_b, max(dists[i] for i in range(len(dists)) if i not in (max_idx, other_idx)))
                if new_max < best_max - 1e-12:
                    routes[max_idx] = new_route_a
                    routes[other_idx] = new_route_b
                    dists[max_idx] = new_dist_a
                    dists[other_idx] = new_dist_b
                    best_max = new_max
                    if best_max < max(route_distance(r) for r in best_routes):
                        best_routes[:] = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                    return True
        return False

    def improve_by_swap(routes, dists, best_max, best_routes):
        # find the two routes with largest distances, but mainly swap involving longest
        max_idx = max(range(len(dists)), key=lambda i: dists[i])
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            return False
        max_custs = max_route[1:-1]
        for other_idx in range(len(routes)):
            if other_idx == max_idx:
                continue
            other_route = routes[other_idx]
            if len(other_route) <= 2:
                continue
            other_custs = other_route[1:-1]
            for cust_a in max_custs:
                for cust_b in other_custs:
                    new_route_a = [0] + [c for c in max_route[1:-1] if c != cust_a] + [cust_b] + [0]
                    new_route_b = [0] + [c for c in other_route[1:-1] if c != cust_b] + [cust_a] + [0]
                    new_route_a = two_opt(new_route_a)
                    new_route_b = two_opt(new_route_b)
                    new_dist_a = route_distance(new_route_a)
                    new_dist_b = route_distance(new_route_b)
                    old_max = max(dists)
                    new_max = max(new_dist_a, new_dist_b, max(dists[i] for i in range(len(dists)) if i not in (max_idx, other_idx)))
                    if new_max < best_max - 1e-12:
                        routes[max_idx] = new_route_a
                        routes[other_idx] = new_route_b
                        dists[max_idx] = new_dist_a
                        dists[other_idx] = new_dist_b
                        best_max = new_max
                        if best_max < max(route_distance(r) for r in best_routes):
                            best_routes[:] = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                        return True
        return False

    best_routes = None
    best_max = float('inf')
    max_restarts = min(truck_count, 5)
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        dists = compute_route_distances(routes)
        current_max = max(dists)
        if current_max < best_max - 1e-12:
            best_routes = [r[:] for r in routes]
            best_max = current_max
            report_best_vrp(best_routes)
        improved = True
        max_iters = n * truck_count
        iters = 0
        while improved and iters < max_iters:
            improved = False
            # try relocate from longest route first
            if improve_by_relocate(routes, dists, best_max, best_routes):
                improved = True
                continue
            # try swap
            if improve_by_swap(routes, dists, best_max, best_routes):
                improved = True
                continue
            iters += 1
    return best_routes
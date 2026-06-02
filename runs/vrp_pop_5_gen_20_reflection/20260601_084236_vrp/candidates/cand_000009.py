import numpy as np
import heapq
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])
        for _ in range(truck_count - (n-1)):
            routes.append([0, 0])
        return routes

    # Helper functions
    def route_distance(route):
        if len(route) == 2:
            return distance_matrix[0, 0]
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def nearest_neighbor_tour(cluster):
        if not cluster:
            return [0, 0]
        unvisited = set(cluster)
        current = 0
        tour = [0]
        while unvisited:
            next_cust = min(unvisited, key=lambda c: distance_matrix[current, c])
            tour.append(next_cust)
            unvisited.remove(next_cust)
            current = next_cust
        tour.append(0)
        return tour

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j-i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        return route

    # Farthest-first seed selection
    seeds = []
    # first seed: farthest from depot
    depot_dist = [(distance_matrix[0, i], i) for i in customers]
    if depot_dist:
        farthest = max(depot_dist, key=lambda x: x[0])[1]
        seeds.append(farthest)
    while len(seeds) < truck_count and len(seeds) < len(customers):
        # for each remaining customer, find min distance to any seed
        best_dist = -1
        best_cust = None
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c, s] for s in seeds)
            if min_dist > best_dist:
                best_dist = min_dist
                best_cust = c
        if best_cust is not None:
            seeds.append(best_cust)
        else:
            break
    # If still not enough seeds, fill with first customers
    for c in customers:
        if c not in seeds:
            seeds.append(c)
            if len(seeds) == truck_count:
                break
    while len(seeds) < truck_count:
        seeds.append(customers[0])

    clusters = [[] for _ in range(truck_count)]
    for idx, seed in enumerate(seeds):
        clusters[idx].append(seed)
    remaining = [c for c in customers if c not in seeds]
    for cust in remaining:
        best_dist = float('inf')
        best_cluster = 0
        for idx, seed in enumerate(seeds):
            d = distance_matrix[cust, seed]
            if d < best_dist:
                best_dist = d
                best_cluster = idx
        clusters[best_cluster].append(cust)

    # Build initial tours
    routes = []
    for cluster in clusters:
        route = nearest_neighbor_tour(cluster)
        route = two_opt(route)
        routes.append(route)
    best_routes = [r[:] for r in routes]
    best_max = max_distance(routes)

    # Report initial best
    try:
        from reporting import report_best_vrp
        report_best_vrp(best_routes)
    except ImportError:
        pass

    n_customers = len(customers)
    max_non_improve = n_customers  # adaptive stopping criterion
    # combined local search: move and swap in same loop
    iter_count = 0
    non_improve = 0
    while non_improve < max_non_improve:
        improved = False
        # try move operations
        for truck_a in range(truck_count):
            route_a = routes[truck_a]
            if len(route_a) <= 2:
                continue
            custs_a = route_a[1:-1]
            for cust in custs_a:
                new_route_a = [0] + [c for c in route_a[1:-1] if c != cust] + [0]
                for truck_b in range(truck_count):
                    if truck_b == truck_a:
                        continue
                    route_b = routes[truck_b]
                    best_insert = None
                    best_dist = float('inf')
                    for i in range(1, len(route_b)):
                        new_route_b = route_b[:i] + [cust] + route_b[i:]
                        d = route_distance(new_route_b)
                        if d < best_dist:
                            best_dist = d
                            best_insert = i
                    if best_insert is None:
                        continue
                    new_route_b = route_b[:best_insert] + [cust] + route_b[best_insert:]
                    old_max = max(route_distance(route_a), route_distance(route_b))
                    other_max = max(route_distance(r) for idx, r in enumerate(routes) if idx not in (truck_a, truck_b))
                    new_max = max(old_max, other_max)  # careful: old_max already includes both
                    new_max = max(route_distance(new_route_a), route_distance(new_route_b), other_max)
                    if new_max < best_max:
                        routes[truck_a] = two_opt(new_route_a)
                        routes[truck_b] = two_opt(new_route_b)
                        best_max = new_max
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            best_routes = [r[:] for r in routes]
            best_max = max_distance(routes)
            try:
                report_best_vrp(best_routes)
            except ImportError:
                pass
            non_improve = 0
        else:
            # try swap operations
            for truck_a in range(truck_count):
                route_a = routes[truck_a]
                if len(route_a) <= 2:
                    continue
                custs_a = route_a[1:-1]
                for truck_b in range(truck_a+1, truck_count):
                    route_b = routes[truck_b]
                    if len(route_b) <= 2:
                        continue
                    custs_b = route_b[1:-1]
                    for cust_a in custs_a:
                        for cust_b in custs_b:
                            new_route_a = [0] + [c for c in route_a[1:-1] if c != cust_a] + [cust_b] + [0]
                            new_route_b = [0] + [c for c in route_b[1:-1] if c != cust_b] + [cust_a] + [0]
                            new_route_a = two_opt(new_route_a)
                            new_route_b = two_opt(new_route_b)
                            other_max = max(route_distance(r) for idx, r in enumerate(routes) if idx not in (truck_a, truck_b))
                            new_max = max(route_distance(new_route_a), route_distance(new_route_b), other_max)
                            if new_max < best_max:
                                routes[truck_a] = new_route_a
                                routes[truck_b] = new_route_b
                                best_max = new_max
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                best_routes = [r[:] for r in routes]
                best_max = max_distance(routes)
                try:
                    report_best_vrp(best_routes)
                except ImportError:
                    pass
                non_improve = 0
            else:
                non_improve += 1
        iter_count += 1
        if iter_count > n_customers * truck_count * 2:  # safety bound
            break

    return best_routes
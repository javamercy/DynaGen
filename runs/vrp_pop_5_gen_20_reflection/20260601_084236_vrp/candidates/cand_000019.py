import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])
        for _ in range(truck_count - (n - 1)):
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
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
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        return route

    def build_solution(first_seed_idx):
        # farthest seed selection with given first seed
        selected = [customers[first_seed_idx]]
        seeds = [selected[0]]
        while len(seeds) < truck_count:
            best_dist = -1
            best_c = None
            for c in customers:
                if c in selected:
                    continue
                min_dist = min(distance_matrix[c, s] for s in selected)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_c = c
            if best_c is None:
                break
            selected.append(best_c)
            seeds.append(best_c)
        # if not enough seeds, fill with remaining customers (distinct)
        while len(seeds) < truck_count:
            for c in customers:
                if c not in seeds:
                    seeds.append(c)
                    break
        # cluster assignment
        clusters = [[] for _ in range(truck_count)]
        for idx, s in enumerate(seeds):
            clusters[idx].append(s)
        for c in customers:
            if c in seeds:
                continue
            best_dist = float('inf')
            best_cluster = 0
            for idx, s in enumerate(seeds):
                d = distance_matrix[c, s]
                if d < best_dist:
                    best_dist = d
                    best_cluster = idx
            clusters[best_cluster].append(c)
        # build tours
        routes = []
        for cluster in clusters:
            route = nearest_neighbor_tour(cluster)
            route = two_opt(route)
            routes.append(route)
        # local search: move and swap
        n_customers = len(customers)
        max_iters = n_customers * truck_count * 2
        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        for _ in range(max_iters):
            improved = False
            # move
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
                        new_max = max(route_distance(new_route_a), route_distance(new_route_b),
                                      max(route_distance(r) for idx, r in enumerate(routes) if idx not in [truck_a, truck_b]))
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
            if not improved:
                break
            current_max = max_distance(routes)
            if current_max < best_max:
                best_routes = [r[:] for r in routes]
                best_max = current_max
        # swap
        for _ in range(max_iters):
            improved = False
            for truck_a in range(truck_count):
                route_a = routes[truck_a]
                if len(route_a) <= 2:
                    continue
                custs_a = route_a[1:-1]
                for truck_b in range(truck_a + 1, truck_count):
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
                            old_max = max(route_distance(route_a), route_distance(route_b))
                            new_max = max(route_distance(new_route_a), route_distance(new_route_b),
                                          max(route_distance(r) for idx, r in enumerate(routes) if idx not in [truck_a, truck_b]))
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
            if not improved:
                break
            current_max = max_distance(routes)
            if current_max < best_max:
                best_routes = [r[:] for r in routes]
                best_max = current_max
        return best_routes, best_max

    # Multi-start with different first seed
    num_restarts = min(truck_count, 5)
    best_routes = None
    best_max = float('inf')
    for restart in range(num_restarts):
        first_seed_idx = restart % len(customers) if customers else 0
        routes, curr_max = build_solution(first_seed_idx)
        if curr_max < best_max:
            best_max = curr_max
            best_routes = [r[:] for r in routes]
    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes
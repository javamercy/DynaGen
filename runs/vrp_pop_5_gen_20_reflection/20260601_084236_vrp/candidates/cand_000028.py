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

    def local_search(routes, best_routes, best_max):
        improved = False
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
                        if d < best_dist - 1e-12:
                            best_dist = d
                            best_insert = i
                    if best_insert is None:
                        continue
                    new_route_b = route_b[:best_insert] + [cust] + route_b[best_insert:]
                    new_max = max(route_distance(new_route_a), route_distance(new_route_b),
                                  max(route_distance(r) for idx, r in enumerate(routes) if idx not in [truck_a, truck_b]))
                    if new_max < best_max - 1e-12:
                        routes[truck_a] = new_route_a
                        routes[truck_b] = new_route_b
                        best_max = new_max
                        if best_max < max_distance(best_routes):
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                        return routes, best_routes, best_max, True
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
                        new_max = max(route_distance(new_route_a), route_distance(new_route_b),
                                      max(route_distance(r) for idx, r in enumerate(routes) if idx not in [truck_a, truck_b]))
                        if new_max < best_max - 1e-12:
                            routes[truck_a] = new_route_a
                            routes[truck_b] = new_route_b
                            best_max = new_max
                            if best_max < max_distance(best_routes):
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            return routes, best_routes, best_max, True
        return routes, best_routes, best_max, False

    def perturb(routes):
        # Only perturb the longest route(s)
        max_dist = max(route_distance(r) for r in routes)
        longest_indices = [i for i, r in enumerate(routes) if abs(route_distance(r) - max_dist) < 1e-12 and len(r) > 3]
        if not longest_indices:
            return routes
        i = random.choice(longest_indices)
        route = routes[i]
        seg_len = random.randint(1, len(route)-2)
        start = random.randint(1, len(route)-seg_len-1)
        removed = route[start:start+seg_len]
        new_route = route[:start] + route[start+seg_len:]
        # Reinsert into any route, greedily to reduce max distance
        for cust in removed:
            best_route_idx = -1
            best_insert_pos = -1
            best_new_max = float('inf')
            for j in range(truck_count):
                r = routes[j]
                for k in range(1, len(r)):
                    candidate_route = r[:k] + [cust] + r[k:]
                    d_candidate = route_distance(candidate_route)
                    # Compute new max distance if we change this route
                    other_max = max(route_distance(rr) for idx, rr in enumerate(routes) if idx != j)
                    new_max = max(d_candidate, other_max)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_route_idx = j
                        best_insert_pos = k
            if best_route_idx == -1:
                # fallback: insert into original route
                routes[i] = new_route[:1] + [cust] + new_route[1:]
            else:
                r = routes[best_route_idx]
                routes[best_route_idx] = r[:best_insert_pos] + [cust] + r[best_insert_pos:]
        # Update the route we removed from (if not already updated)
        routes[i] = new_route
        return routes

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
        # Outer iteration for improvement
        for _ in range(max_outer):
            improved = True
            iters = 0
            max_iters = n * truck_count
            while improved and iters < max_iters:
                routes, best_routes, best_max, improved = local_search(routes, best_routes, best_max)
                iters += 1
            routes = perturb(routes)
            current_max = max_distance(routes)
            if current_max < best_max - 1e-12:
                best_routes = [r[:] for r in routes]
                best_max = current_max
                report_best_vrp(best_routes)
    return best_routes
import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
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
        route = route[:]
        best_dist = route_distance(route)
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new_route)
                    if d < best_dist - 1e-12:
                        best_dist = d
                        route = new_route
                        improved = True
        return route

    def construct_initial(seed):
        random.seed(seed)
        customers = list(range(1, n))
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

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    best_routes = None
    best_max = float('inf')
    max_restarts = max(truck_count, 10)
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        cur_max = max_distance(routes)
        if cur_max < best_max - 1e-12:
            best_routes = [r[:] for r in routes]
            best_max = cur_max
            report_best_vrp(best_routes)
        
        # Intra-route improvement: one pass of 2-opt on each route
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        cur_max = max_distance(routes)
        if cur_max < best_max - 1e-12:
            best_routes = [r[:] for r in routes]
            best_max = cur_max
            report_best_vrp(best_routes)

        max_iters = n * truck_count  # bounded iterations
        for _ in range(max_iters):
            # Find longest route (break ties by smallest index)
            max_dist = -1
            max_idx = -1
            for t in range(truck_count):
                d = route_distance(routes[t])
                if d > max_dist + 1e-12:
                    max_dist = d
                    max_idx = t
                elif abs(d - max_dist) < 1e-12 and t < max_idx:
                    max_idx = t
            max_route = routes[max_idx]
            if len(max_route) <= 2:
                break
            improved = False
            # Relocate from longest route to any other route (first improvement)
            for idx in range(1, len(max_route)-1):
                cust = max_route[idx]
                new_max_route = max_route[:idx] + max_route[idx+1:]
                for t2 in range(truck_count):
                    if t2 == max_idx:
                        continue
                    r2 = routes[t2]
                    for pos in range(1, len(r2)):
                        new_r2 = r2[:pos] + [cust] + r2[pos:]
                        d1 = route_distance(new_max_route)
                        d2 = route_distance(new_r2)
                        other_max = 0.0
                        for t3 in range(truck_count):
                            if t3 not in (max_idx, t2):
                                d = route_distance(routes[t3])
                                if d > other_max:
                                    other_max = d
                        cand_max = max(d1, d2, other_max)
                        if cand_max < best_max - 1e-12:
                            # Apply 2-opt to both modified routes
                            routes[max_idx] = two_opt(new_max_route)
                            routes[t2] = two_opt(new_r2)
                            cur_max = max_distance(routes)
                            if cur_max < best_max - 1e-12:
                                best_routes = [r[:] for r in routes]
                                best_max = cur_max
                                report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Swap between longest route and any other route (first improvement)
            for idx1 in range(1, len(max_route)-1):
                cust1 = max_route[idx1]
                for t2 in range(truck_count):
                    if t2 == max_idx:
                        continue
                    r2 = routes[t2]
                    for idx2 in range(1, len(r2)-1):
                        cust2 = r2[idx2]
                        new_max_route = max_route[:idx1] + [cust2] + max_route[idx1+1:]
                        new_r2 = r2[:idx2] + [cust1] + r2[idx2+1:]
                        d1 = route_distance(new_max_route)
                        d2 = route_distance(new_r2)
                        other_max = 0.0
                        for t3 in range(truck_count):
                            if t3 not in (max_idx, t2):
                                d = route_distance(routes[t3])
                                if d > other_max:
                                    other_max = d
                        cand_max = max(d1, d2, other_max)
                        if cand_max < best_max - 1e-12:
                            routes[max_idx] = two_opt(new_max_route)
                            routes[t2] = two_opt(new_r2)
                            cur_max = max_distance(routes)
                            if cur_max < best_max - 1e-12:
                                best_routes = [r[:] for r in routes]
                                best_max = cur_max
                                report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break  # no more improvements possible
        # End of restart loop
    return best_routes
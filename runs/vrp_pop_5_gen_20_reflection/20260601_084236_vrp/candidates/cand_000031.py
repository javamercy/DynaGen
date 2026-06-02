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
            best = route[:]
            best_dist = route_distance(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new_route)
                    if d < best_dist - 1e-12:
                        best_dist = d
                        best = new_route
                        improved = True
                if improved:
                    break
            route = best
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
    max_restarts = max(truck_count, 10)
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        cur_max = max_distance(routes)
        if cur_max < best_max - 1e-12:
            best_routes = [r[:] for r in routes]
            best_max = cur_max
            report_best_vrp(best_routes)
        # Iterative improvement targeting longest route
        for it in range(n * truck_count):  # bounded iterations
            # Find longest route indices
            dists = [route_distance(r) for r in routes]
            max_idx = max(range(truck_count), key=lambda i: dists[i])
            max_dist = dists[max_idx]
            # Try relocate: move a customer from longest to another route
            improved = False
            for pos in range(1, len(routes[max_idx])-1):
                cust = routes[max_idx][pos]
                for other in range(truck_count):
                    if other == max_idx:
                        continue
                    # try inserting cust at best position in other route to reduce max distance
                    best_insert = None
                    best_new_max = float('inf')
                    for k in range(1, len(routes[other])):
                        new_route = routes[other][:k] + [cust] + routes[other][k:]
                        new_d = route_distance(new_route)
                        # compute new max distances
                        new_dists = []
                        for idx, r in enumerate(routes):
                            if idx == max_idx:
                                # remove cust from max route
                                new_r = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                                new_dists.append(route_distance(new_r))
                            elif idx == other:
                                new_dists.append(new_d)
                            else:
                                new_dists.append(route_distance(r))
                        candidate_max = max(new_dists)
                        if candidate_max < best_new_max - 1e-12:
                            best_new_max = candidate_max
                            best_insert = k
                    if best_new_max < best_max - 1e-12:
                        # accept
                        new_route_max = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                        new_route_other = routes[other][:best_insert] + [cust] + routes[other][best_insert:]
                        # apply 2-opt
                        new_route_max = two_opt(new_route_max)
                        new_route_other = two_opt(new_route_other)
                        routes[max_idx] = new_route_max
                        routes[other] = new_route_other
                        cur_max = max_distance(routes)
                        if cur_max < best_max - 1e-12:
                            best_max = cur_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Try swap: swap a customer from longest with a customer from another route
            for pos1 in range(1, len(routes[max_idx])-1):
                cust1 = routes[max_idx][pos1]
                for other in range(truck_count):
                    if other == max_idx:
                        continue
                    if len(routes[other]) <= 2:
                        continue
                    for pos2 in range(1, len(routes[other])-1):
                        cust2 = routes[other][pos2]
                        # compute potential new routes
                        new_route_max = routes[max_idx][:pos1] + [cust2] + routes[max_idx][pos1+1:]
                        new_route_other = routes[other][:pos2] + [cust1] + routes[other][pos2+1:]
                        new_dists = []
                        for idx, r in enumerate(routes):
                            if idx == max_idx:
                                new_dists.append(route_distance(new_route_max))
                            elif idx == other:
                                new_dists.append(route_distance(new_route_other))
                            else:
                                new_dists.append(route_distance(r))
                        candidate_max = max(new_dists)
                        if candidate_max < best_max - 1e-12:
                            # apply 2-opt
                            new_route_max = two_opt(new_route_max)
                            new_route_other = two_opt(new_route_other)
                            routes[max_idx] = new_route_max
                            routes[other] = new_route_other
                            cur_max = max_distance(routes)
                            if cur_max < best_max - 1e-12:
                                best_max = cur_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        # After inner loop, also apply intra-route 2-opt to all routes (already done in moves, but ensure)
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
    return best_routes
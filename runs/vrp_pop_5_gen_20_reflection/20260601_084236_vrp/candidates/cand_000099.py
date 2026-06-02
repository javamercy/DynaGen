import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = [[0, i, 0] for i in range(1, n)]
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

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        best = route[:]
        best_d = route_distance(route)
        max_iter = n * 5
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new)
                    if d < best_d - 1e-12:
                        best_d = d
                        best = new
                        improved = True
            route = best
            iter_count += 1
        return best

    def construct_initial(seed):
        random.seed(seed)
        # assign customers randomly to trucks
        truck_customers = [[] for _ in range(truck_count)]
        for c in customers:
            truck_customers[random.randrange(truck_count)].append(c)
        routes = []
        for cluster in truck_customers:
            if not cluster:
                routes.append([0, 0])
            else:
                unvisited = set(cluster)
                current = 0
                tour = [0]
                while unvisited:
                    next_c = min(unvisited, key=lambda c: distance_matrix[current, c])
                    tour.append(next_c)
                    unvisited.remove(next_c)
                    current = next_c
                tour.append(0)
                routes.append(two_opt(tour))
        return routes

    best_routes = None
    best_max = float('inf')
    best_total = float('inf')
    max_restarts = max(truck_count * 5, 20)
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        cur_max = max_distance(routes)
        cur_total = total_distance(routes)
        if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
            best_max = cur_max
            best_total = cur_total
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Inter-route relocation (best improvement) - reduce max distance
        improved = True
        max_iter_improve = n * 2
        iter_count = 0
        while improved and iter_count < max_iter_improve:
            improved = False
            best_improv = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            for t1 in range(truck_count):
                r1 = routes[t1]
                for idx in range(1, len(r1)-1):
                    cust = r1[idx]
                    new_r1 = r1[:idx] + r1[idx+1:]
                    for t2 in range(truck_count):
                        if t2 == t1:
                            continue
                        r2 = routes[t2]
                        for pos in range(1, len(r2)):
                            new_r2 = r2[:pos] + [cust] + r2[pos:]
                            d1 = route_distance(new_r1)
                            d2 = route_distance(new_r2)
                            other_max = 0.0
                            other_total = 0.0
                            for idx2, r in enumerate(routes):
                                d = route_distance(r)
                                other_total += d
                                if idx2 not in (t1, t2):
                                    if d > other_max:
                                        other_max = d
                            cand_max = max(d1, d2, other_max)
                            cand_total = other_total + d1 + d2 - route_distance(r1) - route_distance(r2)
                            if cand_max < best_new_max - 1e-12 or (abs(cand_max - best_new_max) < 1e-12 and cand_total < best_new_total - 1e-12):
                                best_new_max = cand_max
                                best_new_total = cand_total
                                best_improv = (t1, idx, new_r1, t2, pos, new_r2)
            if best_improv is not None and (best_new_max < cur_max - 1e-12 or (abs(best_new_max - cur_max) < 1e-12 and best_new_total < cur_total - 1e-12)):
                t1, idx, new_r1, t2, pos, new_r2 = best_improv
                routes[t1] = two_opt(new_r1)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                cur_total = total_distance(routes)
                if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                improved = True
            iter_count += 1

        # Intra-route 2-opt
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        cur_max = max_distance(routes)
        cur_total = total_distance(routes)
        if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
            best_max = cur_max
            best_total = cur_total
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    # Final 2-opt on best routes
    if best_routes:
        for t in range(truck_count):
            best_routes[t] = two_opt(best_routes[t])
        report_best_vrp(best_routes)

    return best_routes
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

    def two_opt_intra(route):
        if len(route) <= 3:
            return route
        improved = True
        while improved:
            improved = False
            best_route = route[:]
            best_d = route_distance(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new_route)
                    if d < best_d - 1e-12:
                        best_d = d
                        best_route = new_route
                        improved = True
                if improved:
                    break
            route = best_route
        return route

    def construct_initial(seed_order):
        rng = random.Random(seed_order)
        order = customers[:]
        rng.shuffle(order)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in order:
            best_truck = None
            best_max_new = float('inf')
            best_route = None
            for t in range(truck_count):
                r = routes[t]
                best_pos = None
                best_inc = float('inf')
                for pos in range(1, len(r)):
                    inc = distance_matrix[r[pos-1], cust] + distance_matrix[cust, r[pos]] - distance_matrix[r[pos-1], r[pos]]
                    if inc < best_inc - 1e-12:
                        best_inc = inc
                        best_pos = pos
                new_route = r[:best_pos] + [cust] + r[best_pos:]
                new_dist = route_distance(new_route)
                current_max = max_distance(routes)
                other_dists = [route_distance(routes[i]) for i in range(truck_count) if i != t]
                cand_max = max(max(other_dists) if other_dists else 0.0, new_dist)
                if cand_max < best_max_new - 1e-12:
                    best_max_new = cand_max
                    best_truck = t
                    best_route = new_route
            routes[best_truck] = best_route
        for t in range(truck_count):
            routes[t] = two_opt_intra(routes[t])
        return routes

    def local_search(routes, best_routes, best_max_global):
        # intra-route improvement first
        for t in range(truck_count):
            new_route = two_opt_intra(routes[t])
            routes[t] = new_route
        cur_max = max_distance(routes)
        if cur_max < best_max_global - 1e-12:
            best_max_global = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        # inter-route 2-opt* and relocate
        improved = True
        max_iters = n * truck_count
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # 2-opt* for all pairs
            for t1 in range(truck_count):
                for t2 in range(t1+1, truck_count):
                    r1 = routes[t1]
                    r2 = routes[t2]
                    if len(r1) <= 2 or len(r2) <= 2:
                        continue
                    best_pair = None
                    best_pair_max = float('inf')
                    for i in range(1, len(r1)-2):
                        for j in range(1, len(r2)-2):
                            new_r1 = r1[:i+1] + r2[j+1:]
                            new_r2 = r2[:j+1] + r1[i+1:]
                            d1 = route_distance(new_r1)
                            d2 = route_distance(new_r2)
                            # check if max of these two routes reduces compared to current max of these two routes
                            old_pair_max = max(route_distance(r1), route_distance(r2))
                            new_pair_max = max(d1, d2)
                            if new_pair_max < old_pair_max - 1e-12:
                                # optionally, consider improving global max
                                temp_routes = routes[:]
                                temp_routes[t1] = new_r1
                                temp_routes[t2] = new_r2
                                cand_global_max = max_distance(temp_routes)
                                if cand_global_max < best_max_global - 1e-12:
                                    best_max_global = cand_global_max
                                    routes[t1] = new_r1
                                    routes[t2] = new_r2
                                    routes[t1] = two_opt_intra(routes[t1])
                                    routes[t2] = two_opt_intra(routes[t2])
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(best_routes)
                                    improved = True
                                    break
                                elif new_pair_max < best_pair_max - 1e-12:
                                    best_pair_max = new_pair_max
                                    best_pair = (t1, t2, new_r1, new_r2)
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # if no 2-opt* improvement, try relocate
            for t1 in range(truck_count):
                for t2 in range(truck_count):
                    if t1 == t2:
                        continue
                    r1 = routes[t1]
                    r2 = routes[t2]
                    if len(r1) < 3:
                        continue
                    # try moving a customer from r1 to r2
                    for idx in range(1, len(r1)-1):  # skip depot
                        cust = r1[idx]
                        new_r1 = r1[:idx] + r1[idx+1:]
                        # insert into r2 at best position
                        best_inc = float('inf')
                        best_pos = 1
                        for pos in range(1, len(r2)):
                            inc = distance_matrix[r2[pos-1], cust] + distance_matrix[cust, r2[pos]] - distance_matrix[r2[pos-1], r2[pos]]
                            if inc < best_inc - 1e-12:
                                best_inc = inc
                                best_pos = pos
                        new_r2 = r2[:best_pos] + [cust] + r2[best_pos:]
                        d1 = route_distance(new_r1)
                        d2 = route_distance(new_r2)
                        old_max_pair = max(route_distance(r1), route_distance(r2))
                        new_max_pair = max(d1, d2)
                        if new_max_pair < old_max_pair - 1e-12:
                            temp_routes = routes[:]
                            temp_routes[t1] = new_r1
                            temp_routes[t2] = new_r2
                            cand_global_max = max_distance(temp_routes)
                            if cand_global_max < best_max_global - 1e-12:
                                best_max_global = cand_global_max
                                routes[t1] = two_opt_intra(new_r1)
                                routes[t2] = two_opt_intra(new_r2)
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return best_routes, best_max_global

    best_routes = None
    best_max = float('inf')
    max_restarts = max(truck_count, 5)
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        cur_max = max_distance(routes)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        routes, best_max = local_search(routes, best_routes, best_max)
    return best_routes
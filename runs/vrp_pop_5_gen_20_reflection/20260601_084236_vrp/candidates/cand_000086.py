import numpy as np
import random
import time

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

    def two_opt(route, max_iter=None):
        if len(route) <= 3:
            return route
        if max_iter is None:
            max_iter = n * 6
        improved = True
        best = route[:]
        best_d = route_distance(route)
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
                routes.append(two_opt(tour))
        return routes

    best_routes = None
    best_max = float('inf')
    best_total = float('inf')
    max_restarts = max(truck_count * 6, 25)
    start_time = time.time()
    timeout = 295.0
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
        routes = construct_initial(restart)
        # Perturbation: random swap of two customers
        if len(customers) >= 2:
            c1, c2 = random.sample(customers, 2)
            # find routes containing c1 and c2
            for t, r in enumerate(routes):
                if c1 in r:
                    i1 = r.index(c1)
                if c2 in r:
                    i2 = r.index(c2)
            routes[i1] = [x if x != c1 else c2 for x in routes[i1]]
            routes[i2] = [x if x != c2 else c1 for x in routes[i2]]
            # convert back? Actually we need to swap positions. Simpler: swap their positions in the routes.
            for t, r in enumerate(routes):
                for idx, val in enumerate(r):
                    if val == c1:
                        routes[t][idx] = c2
                    elif val == c2:
                        routes[t][idx] = c1
        current_max = max_distance(routes)
        current_total = total_distance(routes)
        if current_max < best_max - 1e-12 or (abs(current_max - best_max) < 1e-12 and current_total < best_total - 1e-12):
            best_max = current_max
            best_total = current_total
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Intra-route 2-opt
        for t in range(truck_count):
            routes[t] = two_opt(routes[t], max_iter=n*6)
        cur_max = max_distance(routes)
        cur_total = total_distance(routes)
        if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
            best_max = cur_max
            best_total = cur_total
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Inter-route 2-opt* best improvement (single pass)
        improved = True
        max_iter_improve = n * 4
        iter_count = 0
        while improved and iter_count < max_iter_improve and time.time() - start_time < timeout:
            improved = False
            best_improv = None
            best_new_max = float('inf')
            best_new_total = float('inf')
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
                            other_total = 0.0
                            for idx, r in enumerate(routes):
                                d = route_distance(r)
                                other_total += d
                                if idx not in (t1, t2):
                                    if d > other_max:
                                        other_max = d
                            cand_max = max(d1, d2, other_max)
                            cand_total = other_total + d1 + d2 - route_distance(r1) - route_distance(r2)
                            if cand_max < best_new_max - 1e-12 or (abs(cand_max - best_new_max) < 1e-12 and cand_total < best_new_total - 1e-12):
                                best_new_max = cand_max
                                best_new_total = cand_total
                                best_improv = (t1, t2, i, j, new_r1, new_r2)
            if best_improv is not None and (best_new_max < cur_max - 1e-12 or (abs(best_new_max - cur_max) < 1e-12 and best_new_total < cur_total - 1e-12)):
                t1, t2, i, j, new_r1, new_r2 = best_improv
                routes[t1] = two_opt(new_r1, max_iter=n*6)
                routes[t2] = two_opt(new_r2, max_iter=n*6)
                cur_max = max_distance(routes)
                cur_total = total_distance(routes)
                if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                improved = True
                iter_count += 1

        # Max-route reduction via relocation
        reduction_attempts = 0
        max_reduction = n
        while reduction_attempts < max_reduction and time.time() - start_time < timeout:
            max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
            max_route = routes[max_idx]
            if len(max_route) <= 2:
                break
            found = False
            for idx in range(1, len(max_route)-1):
                cust = max_route[idx]
                new_max_route = max_route[:idx] + max_route[idx+1:]
                for t2 in range(truck_count):
                    if t2 == max_idx:
                        continue
                    r2 = routes[t2]
                    for pos in range(1, len(r2)):
                        new_r2 = r2[:pos] + [cust] + r2[pos:]
                        d_max_new = route_distance(new_max_route)
                        d2_new = route_distance(new_r2)
                        other_max = 0.0
                        other_total = 0.0
                        for idx2, r in enumerate(routes):
                            d = route_distance(r)
                            other_total += d
                            if idx2 not in (max_idx, t2):
                                if d > other_max:
                                    other_max = d
                        cand_max = max(d_max_new, d2_new, other_max)
                        cand_total = other_total + d_max_new + d2_new - route_distance(max_route) - route_distance(r2)
                        if cand_max < cur_max - 1e-12 or (abs(cand_max - cur_max) < 1e-12 and cand_total < cur_total - 1e-12):
                            routes[max_idx] = two_opt(new_max_route, max_iter=n*6)
                            routes[t2] = two_opt(new_r2, max_iter=n*6)
                            cur_max = max_distance(routes)
                            cur_total = total_distance(routes)
                            if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                                best_max = cur_max
                                best_total = cur_total
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                reduction_attempts += 1
            else:
                break

    if best_routes:
        for t in range(truck_count):
            best_routes[t] = two_opt(best_routes[t], max_iter=n*6)
        report_best_vrp(best_routes)
    return best_routes
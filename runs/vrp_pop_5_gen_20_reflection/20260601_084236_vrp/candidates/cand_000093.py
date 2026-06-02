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

    def construct_balanced(seed_idx):
        random.seed(seed_idx)
        # farthest-first seed selection
        seeds = []
        first_seed = random.choice(customers)
        seeds.append(first_seed)
        while len(seeds) < truck_count:
            max_dist = -1
            farthest = None
            for c in customers:
                if c in seeds:
                    continue
                min_dist_to_seed = min(distance_matrix[c, s] for s in seeds)
                if min_dist_to_seed > max_dist:
                    max_dist = min_dist_to_seed
                    farthest = c
            seeds.append(farthest)
        # assign customers to nearest seed, but with balancing: prefer seed with smaller current total distance
        clusters = [[] for _ in range(truck_count)]
        route_loads = [0.0] * truck_count  # approximate load: number of customers assigned
        for i, s in enumerate(seeds):
            clusters[i].append(s)
            route_loads[i] = 1.0
        remaining = [c for c in customers if c not in seeds]
        for cust in remaining:
            # compute distances to all seeds
            dists = [distance_matrix[cust, s] for s in seeds]
            # choose seed with smallest distance, but tie-break by smallest load
            best_idx = 0
            best_dist = dists[0]
            best_load = route_loads[0]
            for i in range(1, truck_count):
                if dists[i] < best_dist - 1e-12:
                    best_dist = dists[i]
                    best_idx = i
                    best_load = route_loads[i]
                elif abs(dists[i] - best_dist) < 1e-12:
                    if route_loads[i] < best_load:
                        best_dist = dists[i]
                        best_idx = i
                        best_load = route_loads[i]
            clusters[best_idx].append(cust)
            route_loads[best_idx] += 1.0
        # build routes with nearest neighbor within cluster, then 2-opt
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
    max_restarts = min(truck_count * 10, 50)
    for restart in range(max_restarts):
        routes = construct_balanced(restart)
        current_max = max_distance(routes)
        current_total = total_distance(routes)
        if (current_max < best_max - 1e-12) or (abs(current_max - best_max) < 1e-12 and current_total < best_total - 1e-12):
            best_max = current_max
            best_total = current_total
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Intra-route 2-opt
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        cur_max = max_distance(routes)
        cur_total = total_distance(routes)
        if (cur_max < best_max - 1e-12) or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
            best_max = cur_max
            best_total = cur_total
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Inter-route 2-opt* best improvement (bounded iterations)
        improved = True
        max_iter_improve = 3 * n
        iter_count = 0
        while improved and iter_count < max_iter_improve:
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
                            if (cand_max < best_new_max - 1e-12) or (abs(cand_max - best_new_max) < 1e-12 and cand_total < best_new_total - 1e-12):
                                best_new_max = cand_max
                                best_new_total = cand_total
                                best_improv = (t1, t2, i, j, new_r1, new_r2)
            if best_improv is not None and (best_new_max < cur_max - 1e-12 or (abs(best_new_max - cur_max) < 1e-12 and best_new_total < cur_total - 1e-12)):
                t1, t2, i, j, new_r1, new_r2 = best_improv
                routes[t1] = two_opt(new_r1)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                cur_total = total_distance(routes)
                if (cur_max < best_max - 1e-12) or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                improved = True
                iter_count += 1

        # Balancing heuristic: repeatedly relocate from longest to shortest route
        balance_improved = True
        max_balance_iter = n
        balance_iter = 0
        while balance_improved and balance_iter < max_balance_iter:
            balance_improved = False
            longest_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
            shortest_idx = min(range(truck_count), key=lambda t: route_distance(routes[t]))
            if longest_idx == shortest_idx:
                break
            long_route = routes[longest_idx]
            short_route = routes[shortest_idx]
            if len(long_route) <= 2 or len(short_route) < 2:
                break
            best_reloc = None
            best_new_max = cur_max
            best_new_total = cur_total
            for idx in range(1, len(long_route)-1):
                cust = long_route[idx]
                new_long = long_route[:idx] + long_route[idx+1:]
                for pos in range(1, len(short_route)):
                    new_short = short_route[:pos] + [cust] + short_route[pos:]
                    d_long = route_distance(new_long)
                    d_short = route_distance(new_short)
                    other_max = 0.0
                    other_total = 0.0
                    for t, r in enumerate(routes):
                        d = route_distance(r)
                        other_total += d
                        if t not in (longest_idx, shortest_idx):
                            if d > other_max:
                                other_max = d
                    cand_max = max(d_long, d_short, other_max)
                    cand_total = other_total + d_long + d_short - route_distance(long_route) - route_distance(short_route)
                    if (cand_max < best_new_max - 1e-12) or (abs(cand_max - best_new_max) < 1e-12 and cand_total < best_new_total - 1e-12):
                        best_new_max = cand_max
                        best_new_total = cand_total
                        best_reloc = (idx, pos, new_long, new_short)
            if best_reloc is not None:
                idx, pos, new_long, new_short = best_reloc
                routes[longest_idx] = two_opt(new_long)
                routes[shortest_idx] = two_opt(new_short)
                cur_max = max_distance(routes)
                cur_total = total_distance(routes)
                if (cur_max < best_max - 1e-12) or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                balance_improved = True
            balance_iter += 1

        # Perturbation: random swap from max route to a random other route
        non_empty = [t for t in range(truck_count) if len(routes[t]) > 2]
        if len(non_empty) >= 2:
            max_idx = max(non_empty, key=lambda t: route_distance(routes[t]))
            other = random.choice([t for t in non_empty if t != max_idx])
            r_max = routes[max_idx]
            r_other = routes[other]
            if len(r_max) > 2 and len(r_other) > 2:
                i = random.randint(1, len(r_max)-2)
                j = random.randint(1, len(r_other)-2)
                cust_max = r_max[i]
                cust_other = r_other[j]
                new_r_max = r_max[:i] + [cust_other] + r_max[i+1:]
                new_r_other = r_other[:j] + [cust_max] + r_other[j+1:]
                routes[max_idx] = two_opt(new_r_max)
                routes[other] = two_opt(new_r_other)
                cur_max = max_distance(routes)
                cur_total = total_distance(routes)
                if (cur_max < best_max - 1e-12) or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)

        # Perturbation: random segment exchange between two random routes
        if len(non_empty) >= 2:
            t1, t2 = random.sample(non_empty, 2)
            r1 = routes[t1]
            r2 = routes[t2]
            if len(r1) > 4 and len(r2) > 4:
                i = random.randint(1, len(r1)-3)
                j = random.randint(1, len(r2)-3)
                seg1 = r1[i:j]
                seg2 = r2[i:j]
                new_r1 = r1[:i] + seg2 + r1[j:]
                new_r2 = r2[:i] + seg1 + r2[j:]
                routes[t1] = two_opt(new_r1)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                cur_total = total_distance(routes)
                if (cur_max < best_max - 1e-12) or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)

        # Post-improvement: additional inter-route 2-opt* (a few more iterations)
        improved = True
        max_iter_post = n
        iter_count = 0
        while improved and iter_count < max_iter_post:
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
                            if (cand_max < best_new_max - 1e-12) or (abs(cand_max - best_new_max) < 1e-12 and cand_total < best_new_total - 1e-12):
                                best_new_max = cand_max
                                best_new_total = cand_total
                                best_improv = (t1, t2, i, j, new_r1, new_r2)
            if best_improv is not None and (best_new_max < cur_max - 1e-12 or (abs(best_new_max - cur_max) < 1e-12 and best_new_total < cur_total - 1e-12)):
                t1, t2, i, j, new_r1, new_r2 = best_improv
                routes[t1] = two_opt(new_r1)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                cur_total = total_distance(routes)
                if (cur_max < best_max - 1e-12) or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                improved = True
                iter_count += 1

    # Final 2-opt on best routes
    if best_routes:
        for t in range(truck_count):
            best_routes[t] = two_opt(best_routes[t])
        report_best_vrp(best_routes)

    return best_routes
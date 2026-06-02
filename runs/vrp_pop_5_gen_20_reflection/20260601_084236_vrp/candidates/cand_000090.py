import numpy as np
import random

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

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        best = route[:]
        best_d = route_distance(route)
        max_iter = max(5, n // 2)
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
    max_restarts = 10
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        current_max = max_distance(routes)
        current_total = total_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Intra-route 2-opt one pass
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        cur_max = max_distance(routes)
        cur_total = total_distance(routes)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Inter-route 2-opt* best improvement (bounded iterations)
        improved = True
        max_iter_improve = n
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
                                if idx not in (t1, t2):
                                    d = route_distance(r)
                                    if d > other_max:
                                        other_max = d
                                    other_total += d
                            cand_max = max(d1, d2, other_max)
                            cand_total = d1 + d2 + other_total
                            if cand_max < best_new_max - 1e-12 or (abs(cand_max - best_new_max) < 1e-12 and cand_total < best_new_total - 1e-12):
                                best_new_max = cand_max
                                best_new_total = cand_total
                                best_improv = (t1, t2, i, j, new_r1, new_r2)
            if best_improv is not None and (best_new_max < cur_max - 1e-12 or (abs(best_new_max - cur_max) < 1e-12 and best_new_total < cur_total - 1e-12)):
                t1, t2, i, j, new_r1, new_r2 = best_improv
                routes[t1] = two_opt(new_r1)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                cur_total = total_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                improved = True
                iter_count += 1

        # Max-route reduction via relocation (bounded attempts) - accept reductions in current max or same max but lower total
        reduction_attempts = 0
        while reduction_attempts < n:
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
                            if idx2 not in (max_idx, t2):
                                d = route_distance(r)
                                if d > other_max:
                                    other_max = d
                                other_total += d
                        cand_max = max(d_max_new, d2_new, other_max)
                        cand_total = d_max_new + d2_new + other_total
                        if cand_max < cur_max - 1e-12 or (abs(cand_max - cur_max) < 1e-12 and cand_total < cur_total - 1e-12):
                            routes[max_idx] = two_opt(new_max_route)
                            routes[t2] = two_opt(new_r2)
                            cur_max = max_distance(routes)
                            cur_total = total_distance(routes)
                            if cur_max < best_max - 1e-12:
                                best_max = cur_max
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

        # Additional balancing: move from longest to shortest if doesn't increase max and reduces total (bounded attempts)
        balancing_attempts = 0
        max_balancing_attempts = n // 2
        while balancing_attempts < max_balancing_attempts:
            lengths = [route_distance(r) for r in routes]
            max_idx = max(range(truck_count), key=lambda t: lengths[t])
            min_idx = min(range(truck_count), key=lambda t: lengths[t])
            if max_idx == min_idx or lengths[max_idx] - lengths[min_idx] < 1e-12:
                break
            max_route = routes[max_idx]
            if len(max_route) <= 2:
                break
            found = False
            for idx in range(1, len(max_route)-1):
                cust = max_route[idx]
                new_max_route = max_route[:idx] + max_route[idx+1:]
                r_min = routes[min_idx]
                for pos in range(1, len(r_min)):
                    new_r_min = r_min[:pos] + [cust] + r_min[pos:]
                    d_max_new = route_distance(new_max_route)
                    d_min_new = route_distance(new_r_min)
                    other_max = 0.0
                    other_total = 0.0
                    for idx2, r in enumerate(routes):
                        if idx2 not in (max_idx, min_idx):
                            d = route_distance(r)
                            if d > other_max:
                                other_max = d
                            other_total += d
                    cand_max = max(d_max_new, d_min_new, other_max)
                    cand_total = d_max_new + d_min_new + other_total
                    if abs(cand_max - cur_max) < 1e-12 and cand_total < cur_total - 1e-12:
                        routes[max_idx] = two_opt(new_max_route)
                        routes[min_idx] = two_opt(new_r_min)
                        cur_max = max_distance(routes)
                        cur_total = total_distance(routes)
                        if cur_max < best_max - 1e-12:
                            best_max = cur_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                        found = True
                        break
                if found:
                    break
            if found:
                balancing_attempts += 1
            else:
                break

        # Perturbation: random swap between two random routes
        non_empty = [t for t in range(truck_count) if len(routes[t]) > 2]
        if len(non_empty) >= 2:
            t1, t2 = random.sample(non_empty, 2)
            r1 = routes[t1]
            r2 = routes[t2]
            i1 = random.randint(1, len(r1)-2)
            i2 = random.randint(1, len(r2)-2)
            cust1 = r1[i1]
            cust2 = r2[i2]
            new_r1 = r1[:i1] + [cust2] + r1[i1+1:]
            new_r2 = r2[:i2] + [cust1] + r2[i2+1:]
            routes[t1] = two_opt(new_r1)
            routes[t2] = two_opt(new_r2)
            cur_max = max_distance(routes)
            cur_total = total_distance(routes)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)

        # Second perturbation: random relocation from longest to random other
        max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
        max_route = routes[max_idx]
        if len(max_route) > 2:
            cust_idx = random.randint(1, len(max_route)-2)
            cust = max_route[cust_idx]
            new_max_route = max_route[:cust_idx] + max_route[cust_idx+1:]
            others = [t for t in range(truck_count) if t != max_idx]
            if others:
                t2 = random.choice(others)
                r2 = routes[t2]
                pos = random.randint(1, len(r2)-1) if len(r2) > 1 else 1
                new_r2 = r2[:pos] + [cust] + r2[pos:]
                routes[max_idx] = two_opt(new_max_route)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                cur_total = total_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)

    # Final 2-opt on best routes
    if best_routes:
        for t in range(truck_count):
            best_routes[t] = two_opt(best_routes[t])
        best_max = max_distance(best_routes)
        report_best_vrp(best_routes)

    return best_routes
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

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def two_opt(route, max_iter=None):
        if len(route) <= 3:
            return route
        if max_iter is None:
            max_iter = n * 4
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
    max_restarts = max(truck_count * 5, 20)
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        # Random perturbation: relocate a random customer
        if len(customers) > 0:
            # pick a random customer from a route with len>2
            candidates = [(t, i) for t in range(truck_count) for i in range(1, len(routes[t])-1)]
            if candidates:
                t_from, i_from = random.choice(candidates)
                cust = routes[t_from][i_from]
                # remove from its route
                new_route_from = routes[t_from][:i_from] + routes[t_from][i_from+1:]
                # pick a different truck to insert into
                possible_trucks = [t for t in range(truck_count) if t != t_from]
                if possible_trucks:
                    t_to = random.choice(possible_trucks)
                    if len(routes[t_to]) >= 2:
                        pos = random.randint(1, len(routes[t_to])-1)
                        new_route_to = routes[t_to][:pos] + [cust] + routes[t_to][pos:]
                        routes[t_from] = new_route_from
                        routes[t_to] = new_route_to
        current_max = max_distance(routes)
        current_total = total_distance(routes)
        if current_max < best_max - 1e-12 or (abs(current_max - best_max) < 1e-12 and current_total < best_total - 1e-12):
            best_max = current_max
            best_total = current_total
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

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

        # Inter-route 2-opt* best improvement
        improved = True
        max_iter_improve = n * 2
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
                if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                improved = True
                iter_count += 1

        # Max-route reduction via relocation
        reduction_attempts = 0
        max_reduction = n  # increased from n//2
        while reduction_attempts < max_reduction:
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
                            routes[max_idx] = two_opt(new_max_route)
                            routes[t2] = two_opt(new_r2)
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
            best_routes[t] = two_opt(best_routes[t])
        best_max = max_distance(best_routes)
        report_best_vrp(best_routes)

    return best_routes
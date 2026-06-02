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

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        best = route[:]
        best_d = route_distance(route)
        max_iter = n
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
    max_restarts = max(truck_count, 10)
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        # Intra-route 2-opt
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        # Compute metrics
        def max_distance(routes):
            return max(route_distance(r) for r in routes)
        def total_distance(routes):
            return sum(route_distance(r) for r in routes)
        current_max = max_distance(routes)
        current_total = total_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        # Inter-route improvement: relocate from longest route
        improved = True
        reloc_iters = 0
        while improved and reloc_iters < n:
            improved = False
            for _ in range(n):
                max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
                max_route = routes[max_idx]
                if len(max_route) <= 2:
                    break
                # try to relocate each customer from longest route
                found = False
                for idx in range(1, len(max_route)-1):
                    cust = max_route[idx]
                    new_max_route = max_route[:idx] + max_route[idx+1:]
                    d_max_new = route_distance(new_max_route)
                    best_cand_max = None
                    best_cand_total = None
                    best_t2 = None
                    best_pos = None
                    for t2 in range(truck_count):
                        if t2 == max_idx:
                            continue
                        r2 = routes[t2]
                        for pos in range(1, len(r2)):
                            new_r2 = r2[:pos] + [cust] + r2[pos:]
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
                            if best_cand_max is None or cand_max < best_cand_max - 1e-12 or (abs(cand_max - best_cand_max) < 1e-12 and cand_total < best_cand_total - 1e-12):
                                best_cand_max = cand_max
                                best_cand_total = cand_total
                                best_t2 = t2
                                best_pos = pos
                    if best_cand_max is not None:
                        if best_cand_max < current_max - 1e-12 or (abs(best_cand_max - current_max) < 1e-12 and best_cand_total < current_total - 1e-12):
                            # apply best move
                            routes[max_idx] = two_opt(new_max_route)
                            r2 = routes[best_t2]
                            new_r2 = r2[:best_pos] + [cust] + r2[best_pos:]
                            routes[best_t2] = two_opt(new_r2)
                            current_max = max_distance(routes)
                            current_total = total_distance(routes)
                            if current_max < best_max - 1e-12:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            improved = True
                            found = True
                            break
                if found:
                    reloc_iters += 1
                    if reloc_iters >= n:
                        break
                else:
                    break
            # end for _ in range(n)
            # after full pass, break if no improvement
            if not improved:
                break
        # End while improved
    # Final 2-opt on best
    if best_routes:
        for t in range(truck_count):
            best_routes[t] = two_opt(best_routes[t])
        report_best_vrp(best_routes)
    return best_routes
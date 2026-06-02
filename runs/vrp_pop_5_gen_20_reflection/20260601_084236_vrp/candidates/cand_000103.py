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
        max_iter = n * 2
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

    def greedy_insertion(routes, unvisited):
        while unvisited:
            best_cust = None
            best_route_idx = None
            best_pos = None
            best_max = float('inf')
            best_total = float('inf')
            for cust in unvisited:
                for t_idx in range(truck_count):
                    route = routes[t_idx]
                    if len(route) == 2:
                        pos = 1
                        new_route = [0, cust, 0]
                        d_new = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    else:
                        for pos in range(1, len(route)):
                            new_route = route[:pos] + [cust] + route[pos:]
                            d_new = route_distance(new_route)
                    other_max = 0.0
                    other_total = 0.0
                    for idx, r in enumerate(routes):
                        if idx == t_idx:
                            d = d_new
                        else:
                            d = route_distance(r)
                        if d > other_max:
                            other_max = d
                        other_total += d
                    cand_max = max(other_max, d_new)
                    cand_total = other_total
                    if cand_max < best_max - 1e-12 or (abs(cand_max - best_max) < 1e-12 and cand_total < best_total - 1e-12):
                        best_max = cand_max
                        best_total = cand_total
                        best_cust = cust
                        best_route_idx = t_idx
                        best_pos = pos
            if best_cust is not None:
                route = routes[best_route_idx]
                new_route = route[:best_pos] + [best_cust] + route[best_pos:]
                routes[best_route_idx] = new_route
                unvisited.remove(best_cust)
        return routes

    best_routes = None
    best_max = float('inf')
    max_restarts = max(truck_count * 5, 10)
    for restart in range(max_restarts):
        random.seed(restart)
        seeds = random.sample(customers, min(truck_count, len(customers)))
        routes = []
        for s in seeds:
            routes.append([0, s, 0])
        remaining = [c for c in customers if c not in seeds]
        routes = greedy_insertion(routes, remaining)
        # ensure exactly truck_count routes
        while len(routes) < truck_count:
            routes.append([0, 0])
        for t in range(truck_count):
            if len(routes[t]) > 2:
                routes[t] = two_opt(routes[t])
        current_max = max_distance(routes)
        current_total = total_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Improvement loop
        improved_overall = True
        iteration = 0
        max_iter_global = n * 3
        while improved_overall and iteration < max_iter_global:
            improved_overall = False
            iteration += 1
            # Intra-route 2-opt
            for t in range(truck_count):
                routes[t] = two_opt(routes[t])
            cur_max = max_distance(routes)
            cur_total = total_distance(routes)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            # Inter-route 2-opt*
            improved_inter = True
            max_iter_inter = n * 2
            inter_count = 0
            while improved_inter and inter_count < max_iter_inter:
                improved_inter = False
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
                    improved_inter = True
                    improved_overall = True
                inter_count += 1
            # Max-route reduction via relocation (best improvement)
            for _ in range(n):
                max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
                max_route = routes[max_idx]
                if len(max_route) <= 2:
                    break
                best_reloc = None
                best_new_max = float('inf')
                best_new_total = float('inf')
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
                            if cand_max < best_new_max - 1e-12 or (abs(cand_max - best_new_max) < 1e-12 and cand_total < best_new_total - 1e-12):
                                best_new_max = cand_max
                                best_new_total = cand_total
                                best_reloc = (max_idx, t2, idx, pos, new_max_route, new_r2)
                if best_reloc is not None and (best_new_max < cur_max - 1e-12 or (abs(best_new_max - cur_max) < 1e-12 and best_new_total < cur_total - 1e-12)):
                    max_idx, t2, idx, pos, new_max_route, new_r2 = best_reloc
                    routes[max_idx] = two_opt(new_max_route)
                    routes[t2] = two_opt(new_r2)
                    cur_max = max_distance(routes)
                    cur_total = total_distance(routes)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                    improved_overall = True
                else:
                    break
            # Swap operator between routes
            for _ in range(n):
                best_swap = None
                best_new_max = float('inf')
                best_new_total = float('inf')
                for t1 in range(truck_count):
                    r1 = routes[t1]
                    if len(r1) <= 2:
                        continue
                    for idx1 in range(1, len(r1)-1):
                        cust1 = r1[idx1]
                        for t2 in range(t1+1, truck_count):
                            r2 = routes[t2]
                            if len(r2) <= 2:
                                continue
                            for idx2 in range(1, len(r2)-1):
                                cust2 = r2[idx2]
                                new_r1 = r1[:idx1] + [cust2] + r1[idx1+1:]
                                new_r2 = r2[:idx2] + [cust1] + r2[idx2+1:]
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
                                    best_swap = (t1, t2, idx1, idx2, new_r1, new_r2)
                if best_swap is not None and (best_new_max < cur_max - 1e-12 or (abs(best_new_max - cur_max) < 1e-12 and best_new_total < cur_total - 1e-12)):
                    t1, t2, idx1, idx2, new_r1, new_r2 = best_swap
                    routes[t1] = two_opt(new_r1)
                    routes[t2] = two_opt(new_r2)
                    cur_max = max_distance(routes)
                    cur_total = total_distance(routes)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                    improved_overall = True
                else:
                    break

    # Final 2-opt on best routes
    if best_routes:
        for t in range(truck_count):
            best_routes[t] = two_opt(best_routes[t])
        best_max = max_distance(best_routes)
        report_best_vrp(best_routes)
    return best_routes
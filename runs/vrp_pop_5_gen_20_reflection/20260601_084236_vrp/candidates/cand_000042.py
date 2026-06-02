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
            best_route = route[:]
            best_dist = route_distance(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new_route)
                    if d < best_dist - 1e-12:
                        best_dist = d
                        best_route = new_route
                        improved = True
                if improved:
                    break
            route = best_route
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
        # Local search loop
        local_improved = True
        max_iters = n * truck_count
        iters = 0
        while local_improved and iters < max_iters:
            local_improved = False
            # Intra-route 2-opt
            for t in range(truck_count):
                new_route = two_opt(routes[t])
                if route_distance(new_route) < route_distance(routes[t]) - 1e-12:
                    routes[t] = new_route
                    cur_max = max_distance(routes)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                    local_improved = True
            # Inter-route 2-opt*
            improved_inter = True
            while improved_inter:
                improved_inter = False
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
                                for idx, r in enumerate(routes):
                                    if idx not in (t1, t2):
                                        d = route_distance(r)
                                        if d > other_max:
                                            other_max = d
                                cand_max = max(d1, d2, other_max)
                                if cand_max < best_max - 1e-12:
                                    routes[t1] = two_opt(new_r1)
                                    routes[t2] = two_opt(new_r2)
                                    cur_max = max_distance(routes)
                                    if cur_max < best_max - 1e-12:
                                        best_max = cur_max
                                        best_routes = [r[:] for r in routes]
                                        report_best_vrp(best_routes)
                                    improved_inter = True
                                    local_improved = True
                                    break
                            if improved_inter:
                                break
                        if improved_inter:
                            break
                    if improved_inter:
                        break
            # Max-route reduction: iterative relocate and swap
            for _ in range(n):  # multiple attempts
                max_route_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
                max_route = routes[max_route_idx]
                if len(max_route) <= 2:
                    break
                best_cand = None
                best_max_cand = float('inf')
                # Try relocate each customer from max route to other routes
                for idx in range(1, len(max_route)-1):
                    cust = max_route[idx]
                    new_route_max = max_route[:idx] + max_route[idx+1:]
                    for t2 in range(truck_count):
                        if t2 == max_route_idx:
                            continue
                        r2 = routes[t2]
                        for pos in range(1, len(r2)):
                            new_r2 = r2[:pos] + [cust] + r2[pos:]
                            d_max_new = route_distance(new_route_max)
                            d2_new = route_distance(new_r2)
                            other_max = 0.0
                            for idx2, r in enumerate(routes):
                                if idx2 not in (max_route_idx, t2):
                                    d = route_distance(r)
                                    if d > other_max:
                                        other_max = d
                            cand_max = max(d_max_new, d2_new, other_max)
                            if cand_max < best_max_cand - 1e-12:
                                best_max_cand = cand_max
                                best_cand = (max_route_idx, t2, 'relocate', idx, pos, new_route_max, new_r2)
                # Try swap each customer in max route with each customer in other routes
                for idx1 in range(1, len(max_route)-1):
                    cust1 = max_route[idx1]
                    for t2 in range(truck_count):
                        if t2 == max_route_idx:
                            continue
                        r2 = routes[t2]
                        for idx2 in range(1, len(r2)-1):
                            cust2 = r2[idx2]
                            new_route_max = max_route[:idx1] + [cust2] + max_route[idx1+1:]
                            new_r2 = r2[:idx2] + [cust1] + r2[idx2+1:]
                            d_max_new = route_distance(new_route_max)
                            d2_new = route_distance(new_r2)
                            other_max = 0.0
                            for idx3, r in enumerate(routes):
                                if idx3 not in (max_route_idx, t2):
                                    d = route_distance(r)
                                    if d > other_max:
                                        other_max = d
                            cand_max = max(d_max_new, d2_new, other_max)
                            if cand_max < best_max_cand - 1e-12:
                                best_max_cand = cand_max
                                best_cand = (max_route_idx, t2, 'swap', idx1, idx2, new_route_max, new_r2)
                if best_cand is not None and best_max_cand < best_max - 1e-12:
                    # Apply best move
                    idx1, idx2, move_type, pos1, pos2, new_route1, new_route2 = best_cand
                    routes[idx1] = two_opt(new_route1)
                    routes[idx2] = two_opt(new_route2)
                    cur_max = max_distance(routes)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                    local_improved = True
                else:
                    break
            iters += 1
    return best_routes
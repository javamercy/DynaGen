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

    def ruin_recreate(routes, seed):
        # Remove ~10% of customers, then greedily reinsert
        all_customers = []
        for r in routes:
            for c in r[1:-1]:
                all_customers.append(c)
        if len(all_customers) <= 1:
            return routes
        random.seed(seed)
        n_remove = max(1, int(len(all_customers) * 0.1))
        to_remove = set(random.sample(all_customers, min(n_remove, len(all_customers))))
        # Remove from routes
        new_routes = []
        removed_customers = []
        for r in routes:
            new_route = [0]
            for c in r[1:-1]:
                if c in to_remove:
                    removed_customers.append(c)
                else:
                    new_route.append(c)
            new_route.append(0)
            new_routes.append(new_route)
        # Reinsert greedily
        random.shuffle(removed_customers)
        for cust in removed_customers:
            best_route_idx = None
            best_pos = None
            best_max = float('inf')
            best_other_max = float('inf')
            for t_idx, r in enumerate(new_routes):
                for pos in range(1, len(r)):
                    new_route = r[:pos] + [cust] + r[pos:]
                    # Compute new max distance
                    max_d = 0.0
                    for idx2, r2 in enumerate(new_routes):
                        if idx2 == t_idx:
                            d = route_distance(new_route)
                        else:
                            d = route_distance(r2)
                        if d > max_d:
                            max_d = d
                    if max_d < best_max - 1e-12 or (abs(max_d - best_max) < 1e-12 and max_d < best_other_max - 1e-12):
                        best_max = max_d
                        best_other_max = max_d
                        best_route_idx = t_idx
                        best_pos = pos
            if best_route_idx is not None:
                r = new_routes[best_route_idx]
                new_routes[best_route_idx] = r[:best_pos] + [cust] + r[best_pos:]
        # Apply 2-opt to routes that changed
        for t_idx in range(truck_count):
            new_routes[t_idx] = two_opt(new_routes[t_idx])
        return new_routes

    best_routes = None
    best_max = float('inf')
    max_restarts = max(truck_count * 5, 15)
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        current_max = max_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Intra-route 2-opt one pass
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        cur_max = max_distance(routes)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Inter-route 2-opt* best improvement (bounded iterations)
        improved = True
        max_iter_improve = max(n * 2, 10)
        iter_count = 0
        while improved and iter_count < max_iter_improve:
            improved = False
            best_improv = None
            best_new_max = float('inf')
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
                            if cand_max < best_new_max - 1e-12:
                                best_new_max = cand_max
                                best_improv = (t1, t2, i, j, new_r1, new_r2)
            if best_improv is not None and best_new_max < cur_max - 1e-12:
                t1, t2, i, j, new_r1, new_r2 = best_improv
                routes[t1] = two_opt(new_r1)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                improved = True
                iter_count += 1

        # Max-route reduction via best-improvement relocation (with load-balancing tie-breaker)
        reduction_attempts = 0
        while reduction_attempts < n:
            max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
            max_route = routes[max_idx]
            if len(max_route) <= 2:
                break
            best_candidate = None
            best_new_max = float('inf')
            best_receiver_dist = float('inf')
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
                        for idx2, r in enumerate(routes):
                            if idx2 not in (max_idx, t2):
                                d = route_distance(r)
                                if d > other_max:
                                    other_max = d
                        cand_max = max(d_max_new, d2_new, other_max)
                        if cand_max < best_new_max - 1e-12 or (abs(cand_max - best_new_max) < 1e-12 and d2_new < best_receiver_dist - 1e-12):
                            best_new_max = cand_max
                            best_receiver_dist = d2_new
                            best_candidate = (max_idx, t2, idx, pos, new_max_route, new_r2)
            if best_candidate is not None and best_new_max < cur_max - 1e-12:
                max_idx, t2, idx, pos, new_max_route, new_r2 = best_candidate
                routes[max_idx] = two_opt(new_max_route)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                reduction_attempts += 1
            else:
                break

        # Ruin and recreate (replaces old perturbations)
        routes = ruin_recreate(routes, restart*10 + 1)  # deterministic seed per restart
        cur_max = max_distance(routes)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        # Second pass of max-route reduction
        reduction_attempts = 0
        while reduction_attempts < n:
            max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
            max_route = routes[max_idx]
            if len(max_route) <= 2:
                break
            best_candidate = None
            best_new_max = float('inf')
            best_receiver_dist = float('inf')
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
                        for idx2, r in enumerate(routes):
                            if idx2 not in (max_idx, t2):
                                d = route_distance(r)
                                if d > other_max:
                                    other_max = d
                        cand_max = max(d_max_new, d2_new, other_max)
                        if cand_max < best_new_max - 1e-12 or (abs(cand_max - best_new_max) < 1e-12 and d2_new < best_receiver_dist - 1e-12):
                            best_new_max = cand_max
                            best_receiver_dist = d2_new
                            best_candidate = (max_idx, t2, idx, pos, new_max_route, new_r2)
            if best_candidate is not None and best_new_max < cur_max - 1e-12:
                max_idx, t2, idx, pos, new_max_route, new_r2 = best_candidate
                routes[max_idx] = two_opt(new_max_route)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                reduction_attempts += 1
            else:
                break

    # Final 2-opt on best routes
    if best_routes:
        for t in range(truck_count):
            best_routes[t] = two_opt(best_routes[t])
        best_max = max_distance(best_routes)
        report_best_vrp(best_routes)

    return best_routes
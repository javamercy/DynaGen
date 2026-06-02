import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customer_count = n - 1
    if truck_count >= customer_count:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        try:
            report_best_vrp(routes)
        except:
            pass
        return routes

    def compute_distance(route):
        if len(route) <= 2:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    def best_insertion(route, cust):
        best_pos = 1
        best_increase = float('inf')
        for pos in range(1, len(route)):
            prev = route[pos-1]
            next_ = route[pos]
            increase = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
            if increase < best_increase - 1e-9:
                best_increase = increase
                best_pos = pos
        return best_pos, best_increase

    def max_regret_insertion(unassigned, routes, truck_count):
        # For each customer, compute new max distance if inserted into each route's best position
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_regret = -1.0
        best_new_max = float('inf')
        current_dists = [compute_distance(r) for r in routes]
        for cust in unassigned:
            candidate_maxes = []
            for t in range(truck_count):
                route = routes[t]
                if len(route) == 2:
                    pos = 1
                    new_dist = 2 * distance_matrix[0][cust]
                else:
                    pos, _ = best_insertion(route, cust)
                    new_dist = compute_distance(route[:pos] + [cust] + route[pos:])
                other_dists = [current_dists[i] for i in range(truck_count) if i != t]
                new_max = max(new_dist, *other_dists)
                candidate_maxes.append((new_max, t, pos))
            candidate_maxes.sort()
            if len(candidate_maxes) < 2:
                continue
            regret = candidate_maxes[1][0] - candidate_maxes[0][0]
            best_new_max_candidate = candidate_maxes[0][0]
            if regret > best_regret + 1e-9 or (abs(regret - best_regret) < 1e-9 and best_new_max_candidate < best_new_max - 1e-9):
                best_regret = regret
                best_cust = cust
                best_route_idx = candidate_maxes[0][1]
                best_pos = candidate_maxes[0][2]
                best_new_max = best_new_max_candidate
        if best_cust is not None:
            return best_cust, best_route_idx, best_pos
        # Fallback: best insertion by max distance
        best_cust = None
        best_t = -1
        best_p = -1
        best_max = float('inf')
        for cust in unassigned:
            for t in range(truck_count):
                route = routes[t]
                if len(route) == 2:
                    pos = 1
                    new_dist = 2 * distance_matrix[0][cust]
                else:
                    pos, _ = best_insertion(route, cust)
                    new_dist = compute_distance(route[:pos] + [cust] + route[pos:])
                other_dists = [current_dists[i] for i in range(truck_count) if i != t]
                new_max = max(new_dist, *other_dists)
                if new_max < best_max - 1e-9:
                    best_max = new_max
                    best_cust = cust
                    best_t = t
                    best_p = pos
        if best_cust is not None:
            return best_cust, best_t, best_p
        return None, None, None

    def local_search(routes):
        max_iter = 200
        for iteration in range(max_iter):
            improved = False
            # Intra-route 2-opt
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_dist = compute_distance(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_distance(new_route)
                        if new_dist < best_dist - 1e-9:
                            best_dist = new_dist
                            best_route = new_route
                if best_dist < compute_distance(routes[t]) - 1e-9:
                    routes[t] = best_route
                    improved = True
            if improved:
                continue
            # Inter-route relocate (minimize max)
            best_move = None
            best_reduction = 0.0
            distances = [compute_distance(r) for r in routes]
            max_dist = max(distances)
            for src in range(truck_count):
                src_route = routes[src]
                if len(src_route) <= 2:
                    continue
                for cust_idx in range(1, len(src_route)-1):
                    cust = src_route[cust_idx]
                    new_src = src_route[:cust_idx] + src_route[cust_idx+1:]
                    for dst in range(truck_count):
                        if dst == src:
                            continue
                        dst_route = routes[dst]
                        if len(dst_route) == 2:
                            pos = 1
                            increase = 2 * distance_matrix[0][cust]
                        else:
                            pos, increase = best_insertion(dst_route, cust)
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        new_dist_src = compute_distance(new_src)
                        new_dist_dst = compute_distance(new_dst)
                        other_dists = [distances[i] for i in range(truck_count) if i not in (src, dst)]
                        new_max = max(new_dist_src, new_dist_dst, *other_dists)
                        reduction = max_dist - new_max
                        if reduction > best_reduction + 1e-9:
                            best_reduction = reduction
                            best_move = (src, dst, new_src, new_dst)
            if best_reduction > 1e-9:
                src, dst, new_src, new_dst = best_move
                routes[src] = new_src
                routes[dst] = new_dst
                improved = True
                continue
            # Inter-route swap
            best_move = None
            best_reduction = 0.0
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            cust1 = route1[i]
                            cust2 = route2[j]
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            new_dist1 = compute_distance(new_route1)
                            new_dist2 = compute_distance(new_route2)
                            other_dists = [distances[k] for k in range(truck_count) if k not in (t1, t2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            reduction = max_dist - new_max
                            if reduction > best_reduction + 1e-9:
                                best_reduction = reduction
                                best_move = (t1, t2, new_route1, new_route2)
            if best_reduction > 1e-9:
                t1, t2, new_route1, new_route2 = best_move
                routes[t1] = new_route1
                routes[t2] = new_route2
                improved = True
                continue
            if not improved:
                break
        return routes

    def perturbation(routes):
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        worst_idx = distances.index(max_dist)
        unassigned = routes[worst_idx][1:-1]
        new_routes = [r[:] for r in routes]
        new_routes[worst_idx] = [0, 0]
        other_indices = [i for i in range(truck_count) if i != worst_idx]
        for idx in other_indices:
            route = new_routes[idx]
            if len(route) > 2:
                removable = route[1:-1]
                k = max(1, int(len(removable) * 0.4))
                random.shuffle(removable)
                to_remove = removable[:k]
                for cust in to_remove:
                    unassigned.append(cust)
                    route.remove(cust)
                new_routes[idx] = [0] + [c for c in route if c != 0] + [0]
        # Reinsert using max-regret
        while unassigned:
            cust, t_best, pos = max_regret_insertion(unassigned, new_routes, truck_count)
            if cust is None:
                break
            new_routes[t_best] = new_routes[t_best][:pos] + [cust] + new_routes[t_best][pos:]
            unassigned.remove(cust)
        return new_routes

    best_routes = None
    best_max_dist = float('inf')
    num_restarts = 5
    cust_list = list(range(1, n))
    cust_depot_dist = [(distance_matrix[0][c], c) for c in cust_list]
    cust_depot_dist.sort(key=lambda x: (-x[0], x[1]))
    random.seed(42)

    for restart in range(num_restarts):
        seeds = []
        start_idx = restart % customer_count
        first_seed = cust_depot_dist[start_idx][1]
        seeds.append(first_seed)
        remaining = [c for c in cust_list if c != first_seed]
        while len(seeds) < truck_count and remaining:
            max_min_dist = -1
            new_seed = -1
            for cust in remaining:
                min_dist = min(distance_matrix[cust][s] for s in seeds)
                if min_dist > max_min_dist + 1e-9:
                    max_min_dist = min_dist
                    new_seed = cust
            if new_seed != -1:
                seeds.append(new_seed)
                remaining.remove(new_seed)
            else:
                break
        if len(seeds) < truck_count:
            for cust in cust_list:
                if cust not in seeds:
                    seeds.append(cust)
                    if len(seeds) == truck_count:
                        break

        routes = [[0,0] for _ in range(truck_count)]
        unassigned = cust_list[:]
        for t in range(truck_count):
            if t < len(seeds):
                cust = seeds[t]
                routes[t] = [0, cust, 0]
                unassigned.remove(cust)
        # Initial insertion using max-regret
        while unassigned:
            cust, t_best, pos = max_regret_insertion(unassigned, routes, truck_count)
            if cust is None:
                # Fallback to best insertion by max
                best_cust = None
                best_t = -1
                best_p = -1
                best_max = float('inf')
                for c in unassigned:
                    for t in range(truck_count):
                        route = routes[t]
                        if len(route) == 2:
                            pos = 1
                            new_dist = 2 * distance_matrix[0][c]
                        else:
                            pos, _ = best_insertion(route, c)
                            new_dist = compute_distance(route[:pos] + [c] + route[pos:])
                        other_dists = [compute_distance(routes[i]) for i in range(truck_count) if i != t]
                        new_max = max(new_dist, *other_dists)
                        if new_max < best_max - 1e-9:
                            best_max = new_max
                            best_cust = c
                            best_t = t
                            best_p = pos
                if best_cust is not None:
                    cust, t_best, pos = best_cust, best_t, best_p
                else:
                    break
            routes[t_best] = routes[t_best][:pos] + [cust] + routes[t_best][pos:]
            unassigned.remove(cust)
        routes = local_search(routes)
        distances = [compute_distance(r) for r in routes]
        curr_max = max(distances)
        if curr_max < best_max_dist - 1e-9:
            best_max_dist = curr_max
            best_routes = [r[:] for r in routes]
            try:
                report_best_vrp(best_routes)
            except:
                pass

        num_perturb_rounds = 30
        for pert_round in range(num_perturb_rounds):
            new_routes = perturbation(routes)
            new_routes = local_search(new_routes)
            new_distances = [compute_distance(r) for r in new_routes]
            new_max = max(new_distances)
            T = 10.0 * math.exp(-0.1 * pert_round) + 0.1
            if new_max < curr_max or random.random() < math.exp(- (new_max - curr_max) / T):
                routes = new_routes
                curr_max = new_max
                if new_max < best_max_dist - 1e-9:
                    best_max_dist = new_max
                    best_routes = [r[:] for r in new_routes]
                    try:
                        report_best_vrp(best_routes)
                    except:
                        pass
    return best_routes
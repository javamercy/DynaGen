import numpy as np
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

    def best_two_insertions(cust, routes):
        # returns (best_cost, best_route_idx, best_pos, second_best_cost)
        best_cost = float('inf')
        best_route_idx = None
        best_pos = None
        second_best_cost = float('inf')
        for t in range(truck_count):
            route = routes[t]
            if len(route) == 2:
                cost = 2 * distance_matrix[0][cust]
                pos = 1
            else:
                best_increase = float('inf')
                best_insert_pos = None
                for pos_candidate in range(1, len(route)):
                    prev = route[pos_candidate-1]
                    next_ = route[pos_candidate]
                    increase = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
                    if increase < best_increase - 1e-9:
                        best_increase = increase
                        best_insert_pos = pos_candidate
                cost = best_increase
                pos = best_insert_pos
            if cost < best_cost - 1e-9:
                second_best_cost = best_cost
                best_cost = cost
                best_route_idx = t
                best_pos = pos
            elif cost < second_best_cost - 1e-9:
                second_best_cost = cost
        return best_cost, best_route_idx, best_pos, second_best_cost

    def local_search(routes):
        max_iter = 500
        for _ in range(max_iter):
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
            # Inter-route relocate
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
                            best_increase = float('inf')
                            best_pos = None
                            for pos_candidate in range(1, len(dst_route)):
                                prev = dst_route[pos_candidate-1]
                                next_ = dst_route[pos_candidate]
                                inc = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
                                if inc < best_increase - 1e-9:
                                    best_increase = inc
                                    best_pos = pos_candidate
                            increase = best_increase
                            pos = best_pos
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
            # Inter-route 2-opt*
            best_move = None
            best_reduction = 0.0
            for t1 in range(truck_count):
                for t2 in range(t1+1, truck_count):
                    route1 = routes[t1]
                    route2 = routes[t2]
                    if len(route1) <= 2 or len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-2):
                        for j in range(1, len(route2)-2):
                            new_route1 = route1[:i+1] + route2[j+1:]
                            new_route2 = route2[:j+1] + route1[i+1:]
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
        worst_route_idx = distances.index(max_dist)
        worst_route = routes[worst_route_idx]
        unassigned = worst_route[1:-1]
        new_routes = [r[:] for r in routes]
        new_routes[worst_route_idx] = [0, 0]
        other_indices = [i for i in range(truck_count) if i != worst_route_idx]
        for idx in other_indices:
            route = new_routes[idx]
            if len(route) > 2:
                removable = route[1:-1]
                k = max(1, len(removable) // 5)  # 20% removal
                random.shuffle(removable)
                to_remove = removable[:k]
                for cust in to_remove:
                    unassigned.append(cust)
                    route.remove(cust)
                    new_routes[idx] = [0] + [c for c in route if c != 0] + [0]
        # Regret-2 insertion
        while unassigned:
            best_cust = None
            best_regret = -float('inf')
            best_route_idx = None
            best_pos = None
            for cust in unassigned:
                best_cost, best_route, best_pos_candidate, second_best_cost = best_two_insertions(cust, new_routes)
                if second_best_cost == float('inf'):
                    regret = 0
                else:
                    regret = second_best_cost - best_cost
                # Tie-break by smaller best_cost, then by smaller index
                if regret > best_regret + 1e-9 or (abs(regret - best_regret) < 1e-9 and (best_cust is None or cust < best_cust)):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = best_route
                    best_pos = best_pos_candidate
            if best_cust is not None:
                route = new_routes[best_route_idx]
                new_routes[best_route_idx] = route[:best_pos] + [best_cust] + route[best_pos:]
                unassigned.remove(best_cust)
            else:
                break
        return new_routes

    best_routes = None
    best_max_dist = float('inf')
    num_restarts = 10
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
        # Construction using regret-2 insertion
        while unassigned:
            best_cust = None
            best_regret = -float('inf')
            best_route_idx = None
            best_pos = None
            for cust in unassigned:
                best_cost, best_route, best_pos_candidate, second_best_cost = best_two_insertions(cust, routes)
                if second_best_cost == float('inf'):
                    regret = 0
                else:
                    regret = second_best_cost - best_cost
                if regret > best_regret + 1e-9 or (abs(regret - best_regret) < 1e-9 and (best_cust is None or cust < best_cust)):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = best_route
                    best_pos = best_pos_candidate
            if best_cust is not None:
                route = routes[best_route_idx]
                routes[best_route_idx] = route[:best_pos] + [best_cust] + route[best_pos:]
                unassigned.remove(best_cust)
            else:
                break
        routes = local_search(routes)
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        if max_dist < best_max_dist - 1e-9:
            best_max_dist = max_dist
            best_routes = [r[:] for r in routes]
            try:
                report_best_vrp(best_routes)
            except:
                pass

        for _ in range(8):  # increased perturbation rounds
            new_routes = perturbation(routes)
            new_routes = local_search(new_routes)
            new_distances = [compute_distance(r) for r in new_routes]
            new_max = max(new_distances)
            if new_max < best_max_dist - 1e-9:
                best_max_dist = new_max
                best_routes = [r[:] for r in new_routes]
                try:
                    report_best_vrp(best_routes)
                except:
                    pass
            routes = new_routes
    return best_routes
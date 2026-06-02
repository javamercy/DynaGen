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

    def local_search(routes):
        max_iter = 50
        for it in range(max_iter):
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
                            improved = True
                routes[t] = best_route
            # Inter-route relocate (best improvement for max distance)
            distances = [compute_distance(r) for r in routes]
            max_dist = max(distances)
            best_move = None
            best_reduction = 0.0
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
                            pos, _ = best_insertion(dst_route, cust)
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
            if not improved:
                break
        return routes

    def perturbation(routes, rnd_state):
        # Destroy longest route completely and 10% of other routes
        distances = [compute_distance(r) for r in routes]
        worst_idx = distances.index(max(distances))
        unassigned = routes[worst_idx][1:-1]
        new_routes = [r[:] for r in routes]
        new_routes[worst_idx] = [0, 0]
        other_indices = [i for i in range(truck_count) if i != worst_idx]
        for idx in other_indices:
            route = new_routes[idx]
            if len(route) > 2:
                removable = route[1:-1]
                k = max(1, int(len(removable) * 0.1))
                rnd_state.shuffle(removable)
                to_remove = removable[:k]
                for cust in to_remove:
                    unassigned.append(cust)
                    route.remove(cust)
                new_routes[idx] = [0] + [c for c in route if c != 0] + [0]
        # Reinsert using minimax insertion
        while unassigned:
            best_cust = None
            best_route_idx = None
            best_pos = None
            best_max = float('inf')
            for cust in unassigned:
                for t in range(truck_count):
                    route = new_routes[t]
                    if len(route) == 2:
                        pos = 1
                        new_dist = 2 * distance_matrix[0][cust]
                    else:
                        pos, _ = best_insertion(route, cust)
                        new_dist = compute_distance([0] + [cust] + [0] if len(route)==2 else route[:pos] + [cust] + route[pos:])
                    other_dists = [compute_distance(new_routes[i]) for i in range(truck_count) if i != t]
                    new_max_val = max(new_dist, *other_dists)
                    if new_max_val < best_max - 1e-9 or (abs(new_max_val - best_max) < 1e-9 and (best_route_idx is None or t < best_route_idx)):
                        best_max = new_max_val
                        best_cust = cust
                        best_route_idx = t
                        best_pos = pos if len(route) > 2 else 1
            if best_cust is not None:
                route = new_routes[best_route_idx]
                new_routes[best_route_idx] = route[:best_pos] + [best_cust] + route[best_pos:]
                unassigned.remove(best_cust)
            else:
                break
        return new_routes

    best_routes = None
    best_max_dist = float('inf')
    num_restarts = 3
    pert_rounds = 3
    cust_list = list(range(1, n))
    cust_depot_dist = [(distance_matrix[0][c], c) for c in cust_list]
    cust_depot_dist.sort(key=lambda x: (-x[0], x[1]))

    for restart in range(num_restarts):
        rnd_state = random.Random(42 + restart)
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
        while unassigned:
            best_cust = None
            best_route_idx = None
            best_pos = None
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
                    other_dists = [compute_distance(routes[i]) for i in range(truck_count) if i != t]
                    new_max_val = max(new_dist, *other_dists)
                    if new_max_val < best_max - 1e-9 or (abs(new_max_val - best_max) < 1e-9 and (best_route_idx is None or t < best_route_idx)):
                        best_max = new_max_val
                        best_cust = cust
                        best_route_idx = t
                        best_pos = pos if len(route) > 2 else 1
            if best_cust is not None:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [best_cust] + routes[best_route_idx][best_pos:]
                unassigned.remove(best_cust)
            else:
                break
        routes = local_search(routes)
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        if max_dist < best_max_dist - 1e-9 or (abs(max_dist - best_max_dist) < 1e-9 and restart == 0):
            best_max_dist = max_dist
            best_routes = [r[:] for r in routes]
            try:
                report_best_vrp(best_routes)
            except:
                pass

        for pert_round in range(pert_rounds):
            new_routes = perturbation(routes, rnd_state)
            new_routes = local_search(new_routes)
            new_distances = [compute_distance(r) for r in new_routes]
            new_max = max(new_distances)
            if new_max < best_max_dist - 1e-9 or (abs(new_max - best_max_dist) < 1e-9):
                best_max_dist = new_max
                best_routes = [r[:] for r in new_routes]
                try:
                    report_best_vrp(best_routes)
                except:
                    pass
            routes = new_routes
    return best_routes
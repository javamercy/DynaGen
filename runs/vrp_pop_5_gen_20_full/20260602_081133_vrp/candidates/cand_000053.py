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
        max_iter = n * truck_count * 2
        for _ in range(max_iter):
            improved = False
            # Intra-route 2-opt (first improvement)
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_distance(new_route)
                        if new_dist < compute_distance(route) - 1e-9:
                            routes[t] = new_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-route relocate (first improvement)
            for src in range(truck_count):
                for cust_idx in range(1, len(routes[src])-1):
                    cust = routes[src][cust_idx]
                    new_src = routes[src][:cust_idx] + routes[src][cust_idx+1:]
                    for dst in range(truck_count):
                        if dst == src:
                            continue
                        if len(routes[dst]) == 2:
                            pos = 1
                            new_dst = [0, cust, 0]
                        else:
                            pos, _ = best_insertion(routes[dst], cust)
                            new_dst = routes[dst][:pos] + [cust] + routes[dst][pos:]
                        new_dist_src = compute_distance(new_src)
                        new_dist_dst = compute_distance(new_dst)
                        old_max = max(compute_distance(r) for r in routes)
                        new_max = max(new_dist_src, new_dist_dst, *[compute_distance(routes[i]) for i in range(truck_count) if i not in (src, dst)])
                        if new_max < old_max - 1e-9:
                            routes[src] = new_src
                            routes[dst] = new_dst
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-route swap (first improvement)
            for t1 in range(truck_count):
                for i in range(1, len(routes[t1])-1):
                    for t2 in range(t1+1, truck_count):
                        for j in range(1, len(routes[t2])-1):
                            cust1 = routes[t1][i]
                            cust2 = routes[t2][j]
                            new_route1 = routes[t1][:i] + [cust2] + routes[t1][i+1:]
                            new_route2 = routes[t2][:j] + [cust1] + routes[t2][j+1:]
                            new_dist1 = compute_distance(new_route1)
                            new_dist2 = compute_distance(new_route2)
                            old_max = max(compute_distance(r) for r in routes)
                            new_max = max(new_dist1, new_dist2, *[compute_distance(routes[i]) for i in range(truck_count) if i not in (t1, t2)])
                            if new_max < old_max - 1e-9:
                                routes[t1] = new_route1
                                routes[t2] = new_route2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-route 2-opt* (first improvement)
            for t1 in range(truck_count):
                for i in range(1, len(routes[t1])-2):
                    for t2 in range(t1+1, truck_count):
                        for j in range(1, len(routes[t2])-2):
                            new_route1 = routes[t1][:i+1] + routes[t2][j+1:]
                            new_route2 = routes[t2][:j+1] + routes[t1][i+1:]
                            new_dist1 = compute_distance(new_route1)
                            new_dist2 = compute_distance(new_route2)
                            old_max = max(compute_distance(r) for r in routes)
                            new_max = max(new_dist1, new_dist2, *[compute_distance(routes[i]) for i in range(truck_count) if i not in (t1, t2)])
                            if new_max < old_max - 1e-9:
                                routes[t1] = new_route1
                                routes[t2] = new_route2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    def perturbation(routes):
        # Destroy: remove all customers from longest route, plus random subset from others (10% of total customers, at least 1)
        distances = [compute_distance(r) for r in routes]
        worst_idx = distances.index(max(distances))
        worst_route = routes[worst_idx]
        if len(worst_route) <= 2:
            return routes
        removed = worst_route[1:-1]
        new_routes = [r[:] for r in routes]
        new_routes[worst_idx] = [0, 0]
        # Additional random customers from other routes
        other_customers = []
        for t in range(truck_count):
            if t != worst_idx:
                other_customers.extend(routes[t][1:-1])
        random.shuffle(other_customers)
        num_extra = max(1, int(0.1 * customer_count))
        extra = other_customers[:num_extra]
        for cust in extra:
            # find which route contains this customer
            for t in range(truck_count):
                if t == worst_idx:
                    continue
                if cust in new_routes[t]:
                    idx = new_routes[t].index(cust)
                    new_routes[t] = new_routes[t][:idx] + new_routes[t][idx+1:]
                    removed.append(cust)
                    break
        # Reinsert all removed customers using minimax
        unassigned = removed[:]
        while unassigned:
            best_cust = None
            best_route_idx = None
            best_pos = None
            best_max = float('inf')
            for cust in unassigned:
                for t in range(truck_count):
                    if len(new_routes[t]) == 2:
                        pos = 1
                        new_route = [0, cust, 0]
                        new_dist = 2 * distance_matrix[0][cust]
                    else:
                        pos, _ = best_insertion(new_routes[t], cust)
                        new_route = new_routes[t][:pos] + [cust] + new_routes[t][pos:]
                        new_dist = compute_distance(new_route)
                    other_dists = [compute_distance(new_routes[i]) for i in range(truck_count) if i != t]
                    new_max_val = max(new_dist, *other_dists)
                    if new_max_val < best_max - 1e-9 or (abs(new_max_val - best_max) < 1e-9 and (best_route_idx is None or t < best_route_idx)):
                        best_max = new_max_val
                        best_cust = cust
                        best_route_idx = t
                        best_pos = pos if len(new_routes[t]) > 2 else 1
            if best_cust is not None:
                new_routes[best_route_idx] = new_routes[best_route_idx][:best_pos] + [best_cust] + new_routes[best_route_idx][best_pos:]
                unassigned.remove(best_cust)
            else:
                break
        return new_routes

    best_routes = None
    best_max_dist = float('inf')
    num_restarts = 3
    # Prepare farthest-first seed list
    cust_list = list(range(1, n))
    cust_depot_dist = [(distance_matrix[0][c], c) for c in cust_list]
    cust_depot_dist.sort(key=lambda x: (-x[0], x[1]))
    
    for restart in range(num_restarts):
        seeds = []
        start_idx = restart % customer_count
        first_seed = cust_depot_dist[start_idx][1]
        seeds.append(first_seed)
        remaining = [c for c in cust_list if c != first_seed]
        while len(seeds) < truck_count and len(remaining) > 0:
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
                    if len(routes[t]) == 2:
                        pos = 1
                        new_route = [0, cust, 0]
                        new_dist = 2 * distance_matrix[0][cust]
                    else:
                        pos, _ = best_insertion(routes[t], cust)
                        new_route = routes[t][:pos] + [cust] + routes[t][pos:]
                        new_dist = compute_distance(new_route)
                    other_dists = [compute_distance(routes[i]) for i in range(truck_count) if i != t]
                    new_max_val = max(new_dist, *other_dists)
                    if new_max_val < best_max - 1e-9 or (abs(new_max_val - best_max) < 1e-9 and (best_route_idx is None or t < best_route_idx)):
                        best_max = new_max_val
                        best_cust = cust
                        best_route_idx = t
                        best_pos = pos if len(routes[t]) > 2 else 1
            if best_cust is not None:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [best_cust] + routes[best_route_idx][best_pos:]
                unassigned.remove(best_cust)
            else:
                break
        # Local search after construction
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
        # Perturbation rounds (2 times)
        for _ in range(2):
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
import numpy as np

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

    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))

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
            nxt = route[pos]
            increase = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
            if increase < best_increase - 1e-9:
                best_increase = increase
                best_pos = pos
        return best_pos, best_increase

    # Farthest-first seed selection
    seeds = []
    if customer_count > 0:
        max_dist = -1
        seed = -1
        for cust in range(1, n):
            d = distance_matrix[0][cust]
            if d > max_dist + 1e-9:
                max_dist = d
                seed = cust
        seeds.append(seed)
        for _ in range(1, truck_count):
            max_min_dist = -1
            new_seed = -1
            for cust in range(1, n):
                if cust in seeds:
                    continue
                min_dist_to_seeds = min(distance_matrix[cust][s] for s in seeds)
                if min_dist_to_seeds > max_min_dist + 1e-9:
                    max_min_dist = min_dist_to_seeds
                    new_seed = cust
            if new_seed == -1:
                break
            seeds.append(new_seed)
        for cust in range(1, n):
            if len(seeds) >= truck_count:
                break
            if cust not in seeds:
                seeds.append(cust)

    # Assign seeds to routes
    for t in range(min(truck_count, len(seeds))):
        cust = seeds[t]
        routes[t] = [0, cust, 0]
        unassigned.remove(cust)

    # Assign remaining customers using mini-max insertion
    while unassigned:
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_new_max = float('inf')
        for cust in unassigned:
            # Compute current route distances
            current_dists = [compute_distance(r) for r in routes]
            current_max = max(current_dists)
            for t in range(truck_count):
                if len(routes[t]) == 2:
                    pos = 1
                    new_dist = 2 * distance_matrix[0][cust]
                else:
                    pos, _ = best_insertion(routes[t], cust)
                    new_route = routes[t][:pos] + [cust] + routes[t][pos:]
                    new_dist = compute_distance(new_route)
                new_max = max(new_dist, *[current_dists[i] for i in range(truck_count) if i != t])
                if new_max < best_new_max - 1e-9 or (abs(new_max - best_new_max) < 1e-9 and t < best_route_idx):
                    best_new_max = new_max
                    best_cust = cust
                    best_route_idx = t
                    best_pos = pos
        if best_cust is not None:
            routes[best_route_idx] = routes[best_route_idx][:best_pos] + [best_cust] + routes[best_route_idx][best_pos:]
            unassigned.remove(best_cust)
        else:
            break

    distances = [compute_distance(r) for r in routes]
    max_dist = max(distances)
    try:
        report_best_vrp(routes)
    except:
        pass

    # Improvement phase
    max_iter = n * truck_count * 2
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
            distances = [compute_distance(r) for r in routes]
            max_dist = max(distances)
            try:
                report_best_vrp(routes)
            except:
                pass
            continue

        # Inter-route relocate
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
                        new_dst = [0, cust, 0]
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
            distances = [compute_distance(r) for r in routes]
            max_dist = max(distances)
            try:
                report_best_vrp(routes)
            except:
                pass
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
            distances = [compute_distance(r) for r in routes]
            max_dist = max(distances)
            try:
                report_best_vrp(routes)
            except:
                pass
            improved = True
            continue

        # Shake: if no improvement, remove longest route and reinsert
        if not improved:
            # Find longest route (max distance)
            max_route_idx = distances.index(max(distances))
            longest_route = routes[max_route_idx]
            if len(longest_route) <= 2:
                break
            # Remove customers from longest route
            customers_to_reinsert = longest_route[1:-1]
            routes[max_route_idx] = [0, 0]
            # Reinsert using mini-max
            for cust in customers_to_reinsert:
                best_route_idx = None
                best_pos = None
                best_new_max = float('inf')
                current_dists = [compute_distance(r) for r in routes]
                current_max = max(current_dists)
                for t in range(truck_count):
                    if len(routes[t]) == 2:
                        pos = 1
                        new_dist = 2 * distance_matrix[0][cust]
                    else:
                        pos, _ = best_insertion(routes[t], cust)
                        new_route = routes[t][:pos] + [cust] + routes[t][pos:]
                        new_dist = compute_distance(new_route)
                    new_max = max(new_dist, *[current_dists[i] for i in range(truck_count) if i != t])
                    if new_max < best_new_max - 1e-9 or (abs(new_max - best_new_max) < 1e-9 and t < best_route_idx):
                        best_new_max = new_max
                        best_route_idx = t
                        best_pos = pos
                if best_route_idx is not None:
                    routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
                else:
                    # fallback: put back to original
                    routes[max_route_idx] = longest_route
                    break
            else:
                # All reinserted, update distances
                distances = [compute_distance(r) for r in routes]
                new_max = max(distances)
                if new_max < max_dist - 1e-9:
                    max_dist = new_max
                    try:
                        report_best_vrp(routes)
                    except:
                        pass
                    improved = True
                    continue
                else:
                    # revert if not improved
                    routes = old_routes
                    distances = [compute_distance(r) for r in routes]
                    max_dist = max(distances)
                    break

        if not improved:
            break

    return routes
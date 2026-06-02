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
    assigned = [False] * n
    assigned[0] = True
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
            next_ = route[pos]
            increase = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
            if increase < best_increase - 1e-9:
                best_increase = increase
                best_pos = pos
        return best_pos, best_increase

    # Seed selection: farthest-first
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
            if new_seed != -1:
                seeds.append(new_seed)
            else:
                break
        for cust in range(1, n):
            if len(seeds) >= truck_count:
                break
            if cust not in seeds:
                seeds.append(cust)

    # Assign seeds to distinct routes
    for t in range(min(truck_count, len(seeds))):
        cust = seeds[t]
        routes[t] = [0, cust, 0]
        unassigned.remove(cust)

    # Minimax insertion for remaining customers
    while unassigned:
        best_cust = None
        best_route = None
        best_pos = None
        best_max = float('inf')
        for cust in unassigned:
            for t in range(truck_count):
                if len(routes[t]) == 2:
                    new_route = [0, cust, 0]
                    new_dist = 2 * distance_matrix[0][cust]
                else:
                    pos, inc = best_insertion(routes[t], cust)
                    new_route = routes[t][:pos] + [cust] + routes[t][pos:]
                    new_dist = compute_distance(new_route)
                # Compute new max distance if this customer is added
                other_dists = [compute_distance(routes[i]) for i in range(truck_count) if i != t]
                new_max = max(new_dist, *other_dists)
                if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and (best_route is None or t < best_route)):
                    best_max = new_max
                    best_cust = cust
                    best_route = t
                    best_pos = pos if len(routes[t]) > 2 else 1
        if best_cust is not None:
            routes[best_route] = routes[best_route][:best_pos] + [best_cust] + routes[best_route][best_pos:]
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
            distances = [compute_distance(r) for r in routes]
            max_dist = max(distances)
            try:
                report_best_vrp(routes)
            except:
                pass
            improved = True
            continue

        # Inter-route swap (exchange customers)
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

        if not improved:
            break

    return routes
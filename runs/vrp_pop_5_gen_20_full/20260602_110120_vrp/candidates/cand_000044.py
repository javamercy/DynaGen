import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = len(distance_matrix)
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    # Initial routes: each customer on its own route
    routes = [[0, i, 0] for i in range(1, n)]
    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_dist(route):
        if len(route) == 2:
            return 0.0
        d = 0.0
        for k in range(len(route)-1):
            d += distance_matrix[route[k], route[k+1]]
        return d

    dist_cache = {}
    def get_dist(route):
        key = tuple(route)
        if key not in dist_cache:
            dist_cache[key] = route_dist(route)
        return dist_cache[key]

    # Phase 1: Clarke-Wright construction with adaptive penalty
    while len(routes) > truck_count:
        current_max = max(get_dist(r) for r in routes)
        candidates = []
        r_idx = list(range(len(routes)))
        for i in r_idx:
            if len(routes[i]) <= 2:
                continue
            for j in r_idx:
                if i >= j or len(routes[j]) <= 2:
                    continue
                r1 = routes[i]
                r2 = routes[j]
                # merge r1 end to r2 start
                last1 = r1[-2]
                first2 = r2[1]
                savings = distance_matrix[0, last1] + distance_matrix[first2, 0] - distance_matrix[last1, first2]
                new_route = r1[:-1] + r2[1:]
                new_dist = get_dist(r1) + get_dist(r2) - distance_matrix[last1, 0] - distance_matrix[0, first2] + distance_matrix[last1, first2]
                new_max = max(new_dist, *[get_dist(routes[k]) for k in r_idx if k != i and k != j])
                threshold = current_max * 1.1
                if new_dist > threshold:
                    penalized_savings = savings * 0.5
                else:
                    penalized_savings = savings
                candidates.append((new_max, -penalized_savings, i, j, 0, new_route))
                # merge r2 end to r1 start
                last2 = r2[-2]
                first1 = r1[1]
                savings2 = distance_matrix[0, last2] + distance_matrix[first1, 0] - distance_matrix[last2, first1]
                new_route2 = r2[:-1] + r1[1:]
                new_dist2 = get_dist(r2) + get_dist(r1) - distance_matrix[last2, 0] - distance_matrix[0, first1] + distance_matrix[last2, first1]
                new_max2 = max(new_dist2, *[get_dist(routes[k]) for k in r_idx if k != i and k != j])
                if new_dist2 > threshold:
                    penalized_savings2 = savings2 * 0.5
                else:
                    penalized_savings2 = savings2
                candidates.append((new_max2, -penalized_savings2, i, j, 1, new_route2))
        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        best = candidates[0]
        i, j = best[2], best[3]
        new_route = best[5]
        if i > j:
            i, j = j, i
        del routes[j]
        del routes[i]
        routes.append(new_route)
        dist_cache = {}

    report_best_vrp(routes)

    # VND improvement function
    def improve_routes(routes):
        improved = True
        max_iter = n
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            route_dists = [get_dist(r) for r in routes]
            current_max = max(route_dists)
            # Intra-route 2-opt
            for idx in range(len(routes)):
                route = routes[idx]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_dist = route_dists[idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_route = new_route[:]
                            improved = True
                if improved:
                    routes[idx] = best_route
                    dist_cache[tuple(best_route)] = best_dist
                    route_dists[idx] = best_dist
                    break
            if improved:
                continue
            # Inter-route relocate to reduce max
            for cust in range(1, n):
                src_idx = None
                src_pos = None
                for idx, route in enumerate(routes):
                    if cust in route:
                        src_idx = idx
                        src_pos = route.index(cust)
                        break
                if src_idx is None:
                    continue
                src_route = routes[src_idx]
                new_src = src_route[:src_pos] + src_route[src_pos+1:]
                if len(new_src) == 2:
                    new_src = [0, 0]
                new_src_dist = sum(distance_matrix[new_src[k], new_src[k+1]] for k in range(len(new_src)-1))
                for tgt_idx, tgt_route in enumerate(routes):
                    if tgt_idx == src_idx:
                        continue
                    if len(tgt_route) <= 2:
                        new_tgt = [0, cust, 0]
                        new_tgt_dist = distance_matrix[0, cust] + distance_matrix[cust, 0]
                        new_max = max(new_src_dist, new_tgt_dist, *[d for i,d in enumerate(route_dists) if i not in (src_idx, tgt_idx)])
                        if new_max < current_max:
                            routes[src_idx] = new_src
                            routes[tgt_idx] = new_tgt
                            dist_cache[tuple(new_src)] = new_src_dist
                            dist_cache[tuple(new_tgt)] = new_tgt_dist
                            route_dists[src_idx] = new_src_dist
                            route_dists[tgt_idx] = new_tgt_dist
                            improved = True
                            break
                    else:
                        for pos in range(1, len(tgt_route)):
                            new_tgt = tgt_route[:pos] + [cust] + tgt_route[pos:]
                            new_tgt_dist = sum(distance_matrix[new_tgt[k], new_tgt[k+1]] for k in range(len(new_tgt)-1))
                            new_max = max(new_src_dist, new_tgt_dist, *[d for i,d in enumerate(route_dists) if i not in (src_idx, tgt_idx)])
                            if new_max < current_max:
                                routes[src_idx] = new_src
                                routes[tgt_idx] = new_tgt
                                dist_cache[tuple(new_src)] = new_src_dist
                                dist_cache[tuple(new_tgt)] = new_tgt_dist
                                route_dists[src_idx] = new_src_dist
                                route_dists[tgt_idx] = new_tgt_dist
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
            if improved:
                continue
            # Inter-route swap to reduce max
            for cust1 in range(1, n):
                idx1 = None
                pos1 = None
                for idx, route in enumerate(routes):
                    if cust1 in route:
                        idx1 = idx
                        pos1 = route.index(cust1)
                        break
                if idx1 is None:
                    continue
                for cust2 in range(cust1+1, n):
                    idx2 = None
                    pos2 = None
                    for idx, route in enumerate(routes):
                        if cust2 in route and idx != idx1:
                            idx2 = idx
                            pos2 = route.index(cust2)
                            break
                    if idx2 is None:
                        continue
                    # compute new routes
                    route1 = routes[idx1]
                    route2 = routes[idx2]
                    new_route1 = route1[:pos1] + [cust2] + route1[pos1+1:]
                    new_route2 = route2[:pos2] + [cust1] + route2[pos2+1:]
                    new_dist1 = sum(distance_matrix[new_route1[k], new_route1[k+1]] for k in range(len(new_route1)-1))
                    new_dist2 = sum(distance_matrix[new_route2[k], new_route2[k+1]] for k in range(len(new_route2)-1))
                    other_dists = [d for i,d in enumerate(route_dists) if i != idx1 and i != idx2]
                    new_max = max(new_dist1, new_dist2, *other_dists)
                    if new_max < current_max:
                        routes[idx1] = new_route1
                        routes[idx2] = new_route2
                        dist_cache[tuple(new_route1)] = new_dist1
                        dist_cache[tuple(new_route2)] = new_dist2
                        route_dists[idx1] = new_dist1
                        route_dists[idx2] = new_dist2
                        improved = True
                        break
                if improved:
                    break
        return routes

    # Apply initial VND
    routes = improve_routes(routes)
    best_routes = [r[:] for r in routes]
    best_max = max(get_dist(r) for r in routes)
    report_best_vrp(routes)

    # Restart phase with improved perturbation and min-max insertion
    max_restarts = max(1, n // 10)  # increased restart count
    for restart in range(max_restarts):
        # Determine longest routes
        route_dists = [get_dist(r) for r in routes]
        sorted_indices = sorted(range(len(routes)), key=lambda i: route_dists[i], reverse=True)
        # Select up to 3 routes or truck_count whichever smaller
        num_to_perturb = min(3, truck_count)
        # Remove customers with largest distance to depot from the longest routes
        total_cust = n - 1
        remove_count = max(1, total_cust // 3)  # remove about 33%
        removed_customers = []
        # Precompute customer distances to depot
        cust_depot_dist = {c: distance_matrix[0, c] for c in range(1, n)}
        for idx in sorted_indices[:num_to_perturb]:
            route = routes[idx]
            if len(route) <= 3:
                continue
            # Get customers in route (excluding depot)
            customers_in_route = [c for c in route if c != 0]
            # Sort customers by distance to depot descending
            sorted_cust = sorted(customers_in_route, key=lambda c: cust_depot_dist[c], reverse=True)
            # Determine how many to remove from this route
            remaining_remove = remove_count - len(removed_customers)
            if remaining_remove <= 0:
                break
            num_remove = min(remaining_remove, len(sorted_cust))
            if num_remove == 0:
                continue
            removed = sorted_cust[:num_remove]
            removed_customers.extend(removed)
            # Build new route without removed customers
            new_route = [0]
            for c in route[1:-1]:
                if c not in removed:
                    new_route.append(c)
            new_route.append(0)
            if len(new_route) == 2:
                new_route = [0, 0]
            routes[idx] = new_route
            dist_cache.pop(tuple(route), None)
            dist_cache[tuple(new_route)] = get_dist(new_route)
        # Reinsert removed customers using min-max insertion
        for cust in removed_customers:
            best_max_val = float('inf')
            best_location = None
            for idx, route in enumerate(routes):
                if len(route) == 2:
                    # empty route
                    new_route = [0, cust, 0]
                    new_dist = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    new_max = max(new_dist, *[get_dist(routes[i]) for i in range(len(routes)) if i != idx])
                    if new_max < best_max_val:
                        best_max_val = new_max
                        best_location = (idx, None)
                else:
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                        new_max = max(new_dist, *[get_dist(routes[i]) for i in range(len(routes)) if i != idx])
                        if new_max < best_max_val:
                            best_max_val = new_max
                            best_location = (idx, pos)
            if best_location is not None:
                idx, pos = best_location
                route = routes[idx]
                if pos is None:
                    new_route = [0, cust, 0]
                else:
                    new_route = route[:pos] + [cust] + route[pos:]
                routes[idx] = new_route
                dist_cache.pop(tuple(route), None)
                dist_cache[tuple(new_route)] = get_dist(new_route)
        # Apply VND again
        routes = improve_routes(routes)
        cur_max = max(get_dist(r) for r in routes)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    return best_routes
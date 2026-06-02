import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = len(distance_matrix)
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]

    # Initialization: each customer on its own route
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

    def clear_cache_for(routes_subset):
        for r in routes_subset:
            dist_cache.pop(tuple(r), None)

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
                # merge r2 after r1
                last1 = r1[-2]
                first2 = r2[1]
                savings = distance_matrix[0, last1] + distance_matrix[first2, 0] - distance_matrix[last1, first2]
                new_route = r1[:-1] + r2[1:]
                new_dist = get_dist(r1) + get_dist(r2) - distance_matrix[last1, 0] - distance_matrix[0, first2] + distance_matrix[last1, first2]
                new_max = max(new_dist, *[get_dist(routes[k]) for k in r_idx if k != i and k != j])
                threshold = current_max * 1.1
                if new_dist > threshold:
                    penalized = savings * 0.3
                else:
                    penalized = savings
                candidates.append((new_max, -penalized, i, j, 0, new_route))
                # merge r1 after r2
                last2 = r2[-2]
                first1 = r1[1]
                savings2 = distance_matrix[0, last2] + distance_matrix[first1, 0] - distance_matrix[last2, first1]
                new_route2 = r2[:-1] + r1[1:]
                new_dist2 = get_dist(r2) + get_dist(r1) - distance_matrix[last2, 0] - distance_matrix[0, first1] + distance_matrix[last2, first1]
                new_max2 = max(new_dist2, *[get_dist(routes[k]) for k in r_idx if k != i and k != j])
                if new_dist2 > threshold:
                    penalized2 = savings2 * 0.3
                else:
                    penalized2 = savings2
                candidates.append((new_max2, -penalized2, i, j, 1, new_route2))
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
        dist_cache.clear()

    report_best_vrp(routes)

    # VND improvement function
    def improve_routes(routes):
        improved = True
        max_iter = n * truck_count  # bounded
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
                    dist_cache.pop(tuple(route), None)
                    dist_cache[tuple(best_route)] = best_dist
                    break
            if improved:
                continue

            # Intra-route Or-opt: move a segment of length 1, 2, 3
            for idx in range(len(routes)):
                route = routes[idx]
                if len(route) <= 4:
                    continue
                best_route = route[:]
                best_dist = route_dists[idx]
                for seg_len in [1, 2, 3]:
                    for start in range(1, len(route)-1-seg_len):
                        seg = route[start:start+seg_len]
                        remaining = route[:start] + route[start+seg_len:]
                        for insert_pos in range(1, len(remaining)):
                            new_route = remaining[:insert_pos] + seg + remaining[insert_pos:]
                            new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                            if new_dist < best_dist:
                                best_dist = new_dist
                                best_route = new_route[:]
                                improved = True
                if improved:
                    routes[idx] = best_route
                    dist_cache.pop(tuple(route), None)
                    dist_cache[tuple(best_route)] = best_dist
                    break
            if improved:
                continue

            # Inter-route relocate (improved: consider all positions, break on improvement)
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
                    if len(tgt_route) == 2:
                        new_tgt = [0, cust, 0]
                        new_tgt_dist = distance_matrix[0, cust] + distance_matrix[cust, 0]
                        new_max = max(new_src_dist, new_tgt_dist, *[d for i,d in enumerate(route_dists) if i not in (src_idx, tgt_idx)])
                        if new_max < current_max:
                            routes[src_idx] = new_src
                            routes[tgt_idx] = new_tgt
                            dist_cache.pop(tuple(src_route), None)
                            dist_cache.pop(tuple(tgt_route), None)
                            dist_cache[tuple(new_src)] = new_src_dist
                            dist_cache[tuple(new_tgt)] = new_tgt_dist
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
                                dist_cache.pop(tuple(src_route), None)
                                dist_cache.pop(tuple(tgt_route), None)
                                dist_cache[tuple(new_src)] = new_src_dist
                                dist_cache[tuple(new_tgt)] = new_tgt_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue

            # Inter-route swap (cross-exchange of single customers)
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
                        if cust2 in route:
                            idx2 = idx
                            pos2 = route.index(cust2)
                            break
                    if idx2 is None or idx2 == idx1:
                        continue
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
                        dist_cache.pop(tuple(route1), None)
                        dist_cache.pop(tuple(route2), None)
                        dist_cache[tuple(new_route1)] = new_dist1
                        dist_cache[tuple(new_route2)] = new_dist2
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue

            # Inter-route cross-exchange: swap segments of length up to 2
            for i1 in range(len(routes)):
                if len(routes[i1]) <= 3:
                    continue
                for i2 in range(i1+1, len(routes)):
                    if len(routes[i2]) <= 3:
                        continue
                    r1 = routes[i1]
                    r2 = routes[i2]
                    for start1 in range(1, len(r1)-2):
                        for end1 in range(start1, min(start1+2, len(r1)-2)):
                            seg1 = r1[start1:end1+1]
                            for start2 in range(1, len(r2)-2):
                                for end2 in range(start2, min(start2+2, len(r2)-2)):
                                    seg2 = r2[start2:end2+1]
                                    new_r1 = r1[:start1] + seg2 + r1[end1+1:]
                                    new_r2 = r2[:start2] + seg1 + r2[end2+1:]
                                    new_dist1 = sum(distance_matrix[new_r1[k], new_r1[k+1]] for k in range(len(new_r1)-1))
                                    new_dist2 = sum(distance_matrix[new_r2[k], new_r2[k+1]] for k in range(len(new_r2)-1))
                                    new_max = max(new_dist1, new_dist2, *[d for i,d in enumerate(route_dists) if i not in (i1,i2)])
                                    if new_max < current_max:
                                        routes[i1] = new_r1
                                        routes[i2] = new_r2
                                        dist_cache.pop(tuple(r1), None)
                                        dist_cache.pop(tuple(r2), None)
                                        dist_cache[tuple(new_r1)] = new_dist1
                                        dist_cache[tuple(new_r2)] = new_dist2
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Apply initial VND
    routes = improve_routes(routes)
    best_routes = [r[:] for r in routes]
    best_max = max(get_dist(r) for r in routes)
    report_best_vrp(routes)

    # Restart phase
    max_restarts = max(1, n // 5)
    for _ in range(max_restarts):
        route_dists = [get_dist(r) for r in routes]
        # Identify longest route(s) to perturb
        sorted_indices = sorted(range(len(routes)), key=lambda i: route_dists[i], reverse=True)
        # Remove from the longest route a number of customers proportional to its length
        num_remove = max(1, (n-1) // 4)  # remove about 25% of customers
        removed_customers = []
        cust_depot_dist = {c: distance_matrix[0, c] for c in range(1, n)}
        for idx in sorted_indices:
            route = routes[idx]
            if len(route) <= 3:
                continue
            customers = [c for c in route if c != 0]
            # Sort by distance to depot descending
            sorted_cust = sorted(customers, key=lambda c: cust_depot_dist[c], reverse=True)
            # Determine how many to remove from this route
            remaining = num_remove - len(removed_customers)
            if remaining <= 0:
                break
            remove_here = min(remaining, len(sorted_cust))
            if remove_here == 0:
                continue
            removed = sorted_cust[:remove_here]
            removed_customers.extend(removed)
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

        # Reinsert removed customers using min-max insertion with best order (highest cost first)
        # Compute insertion cost (increase in max) for each customer
        insertion_costs = []
        for cust in removed_customers:
            best_inc = float('inf')
            best_loc = None
            for idx, route in enumerate(routes):
                if len(route) == 2:
                    new_dist = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    new_max = max(new_dist, *[get_dist(routes[i]) for i in range(len(routes)) if i != idx])
                    if new_max < best_inc:
                        best_inc = new_max
                        best_loc = (idx, None)
                else:
                    for pos in range(1, len(route)):
                        new_dist = sum(distance_matrix[route[:pos]+[cust]+route[pos:][k], (route[:pos]+[cust]+route[pos:])[k+1]] for k in range(len(route[:pos]+[cust]+route[pos:])-1))
                        # Actually compute directly:
                        new_route_temp = route[:pos] + [cust] + route[pos:]
                        new_dist_temp = sum(distance_matrix[new_route_temp[k], new_route_temp[k+1]] for k in range(len(new_route_temp)-1))
                        new_max = max(new_dist_temp, *[get_dist(routes[i]) for i in range(len(routes)) if i != idx])
                        if new_max < best_inc:
                            best_inc = new_max
                            best_loc = (idx, pos)
            insertion_costs.append((best_inc, cust, best_loc))
        # Sort by insertion cost descending (higher cost first) to be inserted first
        insertion_costs.sort(reverse=True, key=lambda x: x[0])
        for _, cust, loc in insertion_costs:
            if loc is None:
                continue
            idx, pos = loc
            route = routes[idx]
            if pos is None:
                new_route = [0, cust, 0]
            else:
                new_route = route[:pos] + [cust] + route[pos:]
            routes[idx] = new_route
            dist_cache.pop(tuple(route), None)
            dist_cache[tuple(new_route)] = get_dist(new_route)

        # Apply VND
        routes = improve_routes(routes)
        cur_max = max(get_dist(r) for r in routes)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    return best_routes
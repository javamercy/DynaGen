import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = len(distance_matrix)
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_dist(route):
        if len(route) <= 2:
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

    # Phase 1: Clarke-Wright savings construction
    routes = [[0, i, 0] for i in range(1, n)]
    while len(routes) > truck_count:
        candidates = []
        for i in range(len(routes)):
            r1 = routes[i]
            if len(r1) <= 2:
                continue
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if len(r2) <= 2:
                    continue
                # merge r1 end -> r2 start
                last1 = r1[-2]
                first2 = r2[1]
                savings = distance_matrix[0, last1] + distance_matrix[first2, 0] - distance_matrix[last1, first2]
                new_route = r1[:-1] + r2[1:]
                new_dist = get_dist(r1) + get_dist(r2) - distance_matrix[last1, 0] - distance_matrix[0, first2] + distance_matrix[last1, first2]
                candidates.append((new_dist, -savings, i, j, 0, new_route))
                # merge r2 end -> r1 start
                last2 = r2[-2]
                first1 = r1[1]
                savings2 = distance_matrix[0, last2] + distance_matrix[first1, 0] - distance_matrix[last2, first1]
                new_route2 = r2[:-1] + r1[1:]
                new_dist2 = get_dist(r2) + get_dist(r1) - distance_matrix[last2, 0] - distance_matrix[0, first1] + distance_matrix[last2, first1]
                candidates.append((new_dist2, -savings2, i, j, 1, new_route2))
        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
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

    # VND improvement focusing on max distance
    def improve_routes(routes):
        improved = True
        max_iter = n
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            route_dists = [get_dist(r) for r in routes]
            current_max = max(route_dists)

            # Inter-route relocate: move one customer to reduce max
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

            # Inter-route swap: swap two customers to reduce max
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
                        improved = True
                        break
                if improved:
                    break
        return routes

    routes = improve_routes(routes)
    best_routes = [r[:] for r in routes]
    best_max = max(get_dist(r) for r in routes)
    report_best_vrp(routes)

    # Restart: remove 20% of customers from longest routes, reinsert
    max_restarts = 1
    for _ in range(max_restarts):
        route_dists = [get_dist(r) for r in routes]
        # Sort routes by distance descending
        sorted_indices = sorted(range(len(routes)), key=lambda i: route_dists[i], reverse=True)
        num_to_modify = max(1, truck_count // 2)
        modify_indices = sorted_indices[:num_to_modify]
        total_cust = n - 1
        remove_total = max(1, total_cust // 5)
        # Proportional removal counts
        dist_sum = sum(route_dists[i] for i in modify_indices)
        if dist_sum == 0:
            continue
        remove_counts = {}
        assigned = 0
        for idx in modify_indices:
            cnt = int(round(remove_total * route_dists[idx] / dist_sum))
            remove_counts[idx] = cnt
            assigned += cnt
        diff = remove_total - assigned
        if diff > 0:
            for idx in modify_indices:
                if diff <= 0:
                    break
                remove_counts[idx] += 1
                diff -= 1
        elif diff < 0:
            for idx in reversed(modify_indices):
                if diff >= 0:
                    break
                if remove_counts[idx] > 0:
                    remove_counts[idx] -= 1
                    diff += 1
        removed_customers = []
        for idx in modify_indices:
            route = routes[idx]
            cnt = remove_counts[idx]
            if cnt <= 0 or len(route) <= 2:
                continue
            if len(route) - 2 < cnt:
                cnt = len(route) - 2
            start_remove = len(route) - 1 - cnt
            removed = route[start_remove:-1]
            removed_customers.extend(removed)
            new_route = route[:start_remove] + [0]
            if len(new_route) == 2:
                new_route = [0, 0]
            routes[idx] = new_route
            dist_cache.clear()
        # Reinsert using best insertion minimizing max
        for cust in removed_customers:
            best_new_max = float('inf')
            best_total_increase = float('inf')
            best_location = None
            for idx, route in enumerate(routes):
                current_route_dist = get_dist(route)
                other_max = max([get_dist(routes[j]) for j in range(len(routes)) if j != idx])
                if len(route) == 2:
                    new_route = [0, cust, 0]
                    new_dist = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    new_max = max(other_max, new_dist)
                    increase = new_dist - current_route_dist
                    if (new_max < best_new_max) or (new_max == best_new_max and increase < best_total_increase):
                        best_new_max = new_max
                        best_total_increase = increase
                        best_location = (idx, None)
                else:
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                        new_max = max(other_max, new_dist)
                        increase = new_dist - current_route_dist
                        if (new_max < best_new_max) or (new_max == best_new_max and increase < best_total_increase):
                            best_new_max = new_max
                            best_total_increase = increase
                            best_location = (idx, pos)
            if best_location is not None:
                idx, pos = best_location
                route = routes[idx]
                if pos is None:
                    new_route = [0, cust, 0]
                else:
                    new_route = route[:pos] + [cust] + route[pos:]
                routes[idx] = new_route
                dist_cache.clear()
        routes = improve_routes(routes)
        cur_max = max(get_dist(r) for r in routes)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    return best_routes
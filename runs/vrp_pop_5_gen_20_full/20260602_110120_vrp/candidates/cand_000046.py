import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = len(distance_matrix)
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]

    # Helper functions
    def route_dist(route):
        if len(route) == 2:
            return 0.0
        d = 0.0
        for k in range(len(route) - 1):
            d += distance_matrix[route[k], route[k+1]]
        return d

    dist_cache = {}
    def get_dist(route):
        key = tuple(route)
        if key not in dist_cache:
            dist_cache[key] = route_dist(route)
        return dist_cache[key]

    def copy_routes(routes):
        return [r[:] for r in routes]

    # Precompute distances from depot
    depot_dist = distance_matrix[0, :]

    # Regret-2 construction with farthest-depot tie-breaking
    def construct():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_route_idx = None
            best_pos = None
            best_insert_cost = float('inf')
            for cust in unassigned:
                costs = []
                for idx, route in enumerate(routes):
                    if len(route) == 2:
                        new_dist = distance_matrix[0, cust] + distance_matrix[cust, 0]
                        costs.append((new_dist, idx, 1))  # pos=1 means insert after depot
                    else:
                        for pos in range(1, len(route)):
                            new_dist = get_dist(route) - distance_matrix[route[pos-1], route[pos]] + distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]]
                            costs.append((new_dist, idx, pos))
                costs.sort(key=lambda x: x[0])
                # take best and second best
                best = costs[0]
                second = costs[1] if len(costs) > 1 else (float('inf'), None, None)
                regret = second[0] - best[0]
                # tie-breaking: farthest from depot
                dist_to_depot = depot_dist[cust]
                if regret > best_regret or (regret == best_regret and dist_to_depot > depot_dist[best_cust] if best_cust is not None else True):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = best[1]
                    best_pos = best[2]
                    best_insert_cost = best[0]
            # Insert best_cust
            route = routes[best_route_idx]
            if len(route) == 2:
                new_route = [0, best_cust, 0]
            else:
                new_route = route[:best_pos] + [best_cust] + route[best_pos:]
            routes[best_route_idx] = new_route
            dist_cache[tuple(new_route)] = best_insert_cost
            unassigned.remove(best_cust)
        return routes

    routes = construct()
    best_routes = copy_routes(routes)
    best_max = max(get_dist(r) for r in routes)
    report_best_vrp(routes)

    # VND improvement function
    def improve(routes):
        improved = True
        max_iter = n * 10  # bounded
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            current_max = max(get_dist(r) for r in routes)
            # Intra-route 2-opt
            for idx in range(len(routes)):
                route = routes[idx]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_dist = get_dist(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_route = new_route[:]
                if best_dist < get_dist(route):
                    routes[idx] = best_route
                    dist_cache[tuple(best_route)] = best_dist
                    improved = True
                    break
            if improved:
                continue
            # Inter-route relocate
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
                        other_dists = [get_dist(r) for i, r in enumerate(routes) if i not in (src_idx, tgt_idx)]
                        new_max = max(new_src_dist, new_tgt_dist, *other_dists)
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
                            other_dists = [get_dist(r) for i, r in enumerate(routes) if i not in (src_idx, tgt_idx)]
                            new_max = max(new_src_dist, new_tgt_dist, *other_dists)
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
            # Inter-route swap
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
                    other_dists = [get_dist(r) for i, r in enumerate(routes) if i not in (idx1, idx2)]
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
            if improved:
                continue
            # Inter-route cross (2-opt*)
            for idx1 in range(len(routes)):
                r1 = routes[idx1]
                if len(r1) <= 2:
                    continue
                for idx2 in range(idx1+1, len(routes)):
                    r2 = routes[idx2]
                    if len(r2) <= 2:
                        continue
                    for i in range(1, len(r1)-1):
                        for j in range(1, len(r2)-1):
                            new_r1 = r1[:i] + r2[j:]
                            new_r2 = r2[:j] + r1[i:]
                            if new_r1[-1] != 0:
                                new_r1.append(0)
                            if new_r2[-1] != 0:
                                new_r2.append(0)
                            # Ensure they end at depot
                            if new_r1[-1] != 0:
                                continue
                            if new_r2[-1] != 0:
                                continue
                            new_dist1 = sum(distance_matrix[new_r1[k], new_r1[k+1]] for k in range(len(new_r1)-1))
                            new_dist2 = sum(distance_matrix[new_r2[k], new_r2[k+1]] for k in range(len(new_r2)-1))
                            other_dists = [get_dist(r) for i_r, r in enumerate(routes) if i_r not in (idx1, idx2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < current_max:
                                routes[idx1] = new_r1
                                routes[idx2] = new_r2
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
        return routes

    routes = improve(routes)
    cur_max = max(get_dist(r) for r in routes)
    if cur_max < best_max:
        best_max = cur_max
        best_routes = copy_routes(routes)
        report_best_vrp(routes)

    # Restart phase
    max_restarts = max(1, n // 10)
    for restart in range(max_restarts):
        # Identify customers to remove based on marginal contribution
        route_dists = [get_dist(r) for r in routes]
        # Sort routes by current max distance descending
        sorted_indices = sorted(range(len(routes)), key=lambda i: route_dists[i], reverse=True)
        # Decide number of routes to modify: half of trucks
        num_modify = max(1, truck_count // 2)
        modify_indices = sorted_indices[:num_modify]
        # Compute marginal contribution for each customer in selected routes
        marginal_cust = []
        for idx in modify_indices:
            route = routes[idx]
            if len(route) <= 2:
                continue
            for pos in range(1, len(route)-1):
                cust = route[pos]
                # route distance without cust
                new_route = route[:pos] + route[pos+1:]
                new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                marginal = get_dist(route) - new_dist
                marginal_cust.append((marginal, idx, pos, cust))
        # Sort by marginal descending
        marginal_cust.sort(key=lambda x: x[0], reverse=True)
        # Remove up to 20% of customers
        total_cust = n - 1
        remove_count = max(1, total_cust // 5)
        removed = marginal_cust[:remove_count]
        # Remove these customers
        for _, idx, pos, cust in removed:
            route = routes[idx]
            new_route = route[:pos] + route[pos+1:]
            if len(new_route) == 2:
                new_route = [0, 0]
            routes[idx] = new_route
            dist_cache[tuple(route)] = None  # invalidate
            dist_cache[tuple(new_route)] = get_dist(new_route)
        # Reinsert using max-aware best insertion
        removed_custs = [c for _, _, _, c in removed]
        # Shuffle to avoid deterministic order? We'll keep order as they were sorted (deterministic)
        for cust in removed_custs:
            best_new_max = float('inf')
            best_total_increase = float('inf')
            best_location = None
            for idx, route in enumerate(routes):
                current_dist = get_dist(route)
                other_max = max([get_dist(routes[j]) for j in range(len(routes)) if j != idx])
                if len(route) == 2:
                    new_route = [0, cust, 0]
                    new_dist = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    new_max = max(other_max, new_dist)
                    increase = new_dist - current_dist
                    if new_max < best_new_max or (new_max == best_new_max and increase < best_total_increase):
                        best_new_max = new_max
                        best_total_increase = increase
                        best_location = (idx, None)
                else:
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                        new_max = max(other_max, new_dist)
                        increase = new_dist - current_dist
                        if new_max < best_new_max or (new_max == best_new_max and increase < best_total_increase):
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
                dist_cache[tuple(route)] = None
                dist_cache[tuple(new_route)] = get_dist(new_route)
        # Apply VND again
        routes = improve(routes)
        cur_max = max(get_dist(r) for r in routes)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = copy_routes(routes)
            report_best_vrp(routes)
    return best_routes
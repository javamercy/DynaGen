import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = len(distance_matrix)
    # Initial routes: each customer on its own route
    routes = []
    for i in range(1, n):
        routes.append([0, i, 0])
    while len(routes) < truck_count:
        routes.append([0, 0])
    if n == 1:
        return routes

    def route_dist(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    dist_cache = {}
    def get_dist(route):
        key = tuple(route)
        if key not in dist_cache:
            dist_cache[key] = route_dist(route)
        return dist_cache[key]

    while len(routes) > truck_count:
        candidates = []
        r_idx = list(range(len(routes)))
        # Compute current maximum route distance
        current_max = max(get_dist(r) for r in routes)
        for i in r_idx:
            if len(routes[i]) <= 2:
                continue
            for j in r_idx:
                if i >= j or len(routes[j]) <= 2:
                    continue
                r1 = routes[i]
                r2 = routes[j]
                # Merge r1 end to r2 start
                last1 = r1[-2]
                first2 = r2[1]
                savings = distance_matrix[0, last1] + distance_matrix[first2, 0] - distance_matrix[last1, first2]
                new_route = r1[:-1] + r2[1:]
                new_dist = get_dist(r1) + get_dist(r2) - distance_matrix[last1, 0] - distance_matrix[0, first2] + distance_matrix[last1, first2]
                new_max = max(new_dist, *[get_dist(routes[k]) for k in r_idx if k != i and k != j])
                # Set threshold: current_max * 1.1
                threshold = current_max * 1.1
                if new_dist > threshold:
                    penalized_savings = savings * 0.5
                else:
                    penalized_savings = savings
                # Tie-breaking: (max distance, then savings, then indices)
                candidates.append((new_max, -penalized_savings, i, j, 0, last1, first2, new_route))
                # Merge r2 end to r1 start
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
                candidates.append((new_max2, -penalized_savings2, i, j, 1, last2, first1, new_route2))

        if not candidates:
            break
        # Sort by new_max ascending, then by savings descending (higher savings better)
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
        best = candidates[0]
        i, j = best[2], best[3]
        new_route = best[7]
        if i > j:
            i, j = j, i
        del routes[j]
        del routes[i]
        routes.append(new_route)
        dist_cache = {}

    report_best_vrp(routes)

    # Local search improvement
    max_iter = n  # bounded by instance size
    for _ in range(max_iter):
        improved = False
        # Intra-route 2-opt
        for idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = get_dist(best_route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_route = new_route[:]
                        improved = True
            routes[idx] = best_route
            dist_cache[tuple(best_route)] = best_dist

        # Inter-route relocation focused on reducing max
        route_dists = [get_dist(r) for r in routes]
        current_max = max(route_dists)
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
                new_src = [0,0]
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
                        current_max = new_max
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
                            current_max = new_max
                            route_dists[src_idx] = new_src_dist
                            route_dists[tgt_idx] = new_tgt_dist
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        if not improved:
            break

    report_best_vrp(routes)
    return routes
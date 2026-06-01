import numpy as np
import heapq

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    EPS = 1e-12

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - EPS:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Min-max savings construction
    # Each customer starts as a route
    route_list = [[0, c, 0] for c in customers]
    # Maintain endpoints for each route: (first_interior, last_interior)
    endpoints = []
    for r in route_list:
        interior = r[1:-1]
        if interior:
            endpoints.append((interior[0], interior[-1]))
        else:
            endpoints.append((None, None))

    while len(route_list) > truck_count:
        best_new_max = float('inf')
        best_merge = None  # (ri, rj, merged_route)
        # Iterate over all route pairs
        for ri in range(len(route_list)):
            route_i = route_list[ri]
            first_i, last_i = endpoints[ri]
            if first_i is None:
                continue
            for rj in range(ri+1, len(route_list)):
                route_j = route_list[rj]
                first_j, last_j = endpoints[rj]
                if first_j is None:
                    continue
                # Check all four merge possibilities
                candidates = []
                if last_i is not None and first_j is not None:
                    # i end to j start
                    candidates.append((ri, rj, route_i[:-1] + route_j[1:], first_i, last_j))
                if last_j is not None and first_i is not None:
                    # j end to i start
                    candidates.append((rj, ri, route_j[:-1] + route_i[1:], first_j, last_i))
                if first_i is not None and last_j is not None:
                    # i start to j end (reverse of above? actually j end to i start already covers)
                    # Additional possibility: if i is reversed? We only allow merging without reversal.
                    # Standard Clarke-Wright only allows merging at ends without reversal.
                    pass
                # Actually the four cases are covered by two: one where we connect last_i to first_j, one where we connect last_j to first_i.
                # We already have both.
                # Evaluate each candidate
                for ri_idx, rj_idx, merged, new_first, new_last in candidates:
                    # Compute new max distance if we merge these two routes
                    new_routes = [route_list[idx] for idx in range(len(route_list)) if idx not in (ri_idx, rj_idx)]
                    new_routes.append(merged)
                    # Compute max distance of all routes
                    # We can optimize by precomputing distances, but simple approach
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_new_max - EPS:
                        best_new_max = new_max
                        best_merge = (ri_idx, rj_idx, merged, new_first, new_last)
                    elif abs(new_max - best_new_max) <= EPS:
                        # Tie-breaking: prefer smaller customer indices in merged route? Use lexicographic order of merged interior
                        if best_merge is None:
                            best_merge = (ri_idx, rj_idx, merged, new_first, new_last)
                        else:
                            # Compare first interior customer of merged
                            merged_interior = merged[1:-1]
                            best_interior = best_merge[2][1:-1]
                            if merged_interior < best_interior:
                                best_merge = (ri_idx, rj_idx, merged, new_first, new_last)
        if best_merge is None:
            # No feasible merge, should not happen because we can always merge two nonempty routes
            break
        ri, rj, merged, new_first, new_last = best_merge
        # Remove the two old routes and add the merged one
        new_route_list = [route_list[idx] for idx in range(len(route_list)) if idx not in (ri, rj)]
        new_route_list.append(merged)
        route_list = new_route_list
        # Update endpoints
        endpoints = []
        for r in route_list:
            interior = r[1:-1]
            if interior:
                endpoints.append((interior[0], interior[-1]))
            else:
                endpoints.append((None, None))

    # If after construction there are fewer than truck_count routes (should not happen), add empty routes
    while len(route_list) < truck_count:
        route_list.append([0, 0])

    report_best_vrp(route_list)

    # Improvement phase
    max_iter = min(300, n * truck_count)
    for _ in range(max_iter):
        improved = False
        dists = [route_distance(r) for r in route_list]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = list(route_list[max_idx][1:-1])
        if not interior:
            continue

        # First-improvement relocate from longest route
        for cust in interior:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = route_list[other_idx]
                for pos in range(1, len(other_route)):
                    new_routes = [list(r) for r in route_list]
                    idx = new_routes[max_idx].index(cust)
                    new_routes[max_idx].pop(idx)
                    new_routes[other_idx].insert(pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_max - EPS:
                        route_list = new_routes
                        report_best_vrp(route_list)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # First-improvement swap between longest and another
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_interior = list(route_list[other_idx][1:-1])
            if not other_interior:
                continue
            for cust_max in interior:
                for cust_other in other_interior:
                    new_routes = [list(r) for r in route_list]
                    idx_max = new_routes[max_idx].index(cust_max)
                    idx_other = new_routes[other_idx].index(cust_other)
                    new_routes[max_idx][idx_max] = cust_other
                    new_routes[other_idx][idx_other] = cust_max
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_max - EPS:
                        route_list = new_routes
                        report_best_vrp(route_list)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # First-improvement 2-opt on each route
        for idx in range(truck_count):
            route = route_list[idx]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = route_distance(route)
            found = False
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - EPS:
                        best_dist = new_dist
                        best_route = new_route
                        found = True
                        break
                if found:
                    break
            if found:
                route_list[idx] = best_route
                new_max = max(route_distance(r) for r in route_list)
                if new_max < best_max - EPS:
                    report_best_vrp(route_list)
                improved = True
                break
        if not improved:
            break

    return best_routes if best_routes is not None else route_list
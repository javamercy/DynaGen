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
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Adaptive lambda: starts at 1.0, increases as number of routes decreases
    # We'll recalculate savings with current lambda at each merge step
    # But to keep efficient, we can just recompute savings each time? Or compute once and adjust?
    # Simpler: use fixed lambda=1.0 but adapt improvement schedule.
    # Actually, let's implement adaptive lambda: we'll recompute savings each iteration with lambda = 1.0 + (current_routes - truck_count)/truck_count * 0.5
    # This makes lambda larger when there are many routes, encouraging merges with higher savings (since saving = d0i+d0j - lambda*dij).
    # We'll recompute the heap each time routes change, which is O(n^2) but instance sizes are small.

    # Initialize routes
    route_list = [[0, c, 0] for c in customers]
    cust_to_route = {c: idx for idx, c in enumerate(customers)}
    endpoints = [(c, c) for c in customers]

    while len(route_list) > truck_count:
        current_routes = len(route_list)
        lam = 1.0 + (current_routes - truck_count) / truck_count * 0.5
        # Build savings heap
        savings = []
        for i in customers:
            for j in customers:
                if i < j:
                    # Only consider if i and j are endpoints of different routes?
                    # Original uses all pairs, but checks later if merge compatible.
                    # For efficiency, we can skip pairs not endpoints.
                    # Let's only consider endpoints for speed.
                    ri = cust_to_route.get(i)
                    rj = cust_to_route.get(j)
                    if ri is not None and rj is not None and ri != rj:
                        first_i, last_i = endpoints[ri]
                        first_j, last_j = endpoints[rj]
                        if (i == last_i and j == first_j) or (j == last_j and i == first_i) or (i == first_i and j == last_j) or (j == first_j and i == last_i):
                            s = distance_matrix[0, i] + distance_matrix[0, j] - lam * distance_matrix[i, j]
                            heapq.heappush(savings, (-s, i, j))
        if not savings:
            break
        # Process savings
        used = set()
        while len(route_list) > truck_count and savings:
            neg_s, i, j = heapq.heappop(savings)
            if i in used or j in used:
                continue
            ri = cust_to_route.get(i)
            rj = cust_to_route.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            first_i, last_i = endpoints[ri]
            first_j, last_j = endpoints[rj]
            merged = None
            new_first = None
            new_last = None
            if i == last_i and j == first_j:
                merged = route_list[ri][:-1] + route_list[rj][1:]
                new_first = first_i
                new_last = last_j
            elif j == last_j and i == first_i:
                merged = route_list[rj][:-1] + route_list[ri][1:]
                new_first = first_j
                new_last = last_i
            elif i == first_i and j == last_j:
                merged = route_list[rj][:-1] + route_list[ri][1:]
                new_first = first_j
                new_last = last_i
            elif j == first_j and i == last_i:
                merged = route_list[ri][:-1] + route_list[rj][1:]
                new_first = first_i
                new_last = last_j
            else:
                continue
            # Merge
            new_route_list = [r for idx, r in enumerate(route_list) if idx not in (ri, rj)]
            new_route_list.append(merged)
            route_list = new_route_list
            # Update data structures
            # Rebuild cust_to_route and endpoints
            cust_to_route.clear()
            endpoints.clear()
            for idx, r in enumerate(route_list):
                interior = r[1:-1]
                for c in interior:
                    cust_to_route[c] = idx
                first_c = interior[0] if interior else None
                last_c = interior[-1] if interior else None
                endpoints.append((first_c, last_c))
            # Mark i and j as used
            used.add(i)
            used.add(j)
            # Recompute savings? Actually we are still in the heap, but we broke out of inner while? 
            # For simplicity, after each merge we break and rebuild savings in outer while.
            break  # break to rebuild savings with updated routes

    # If still too many routes, merge smallest by distance (same as parent)
    while len(route_list) > truck_count:
        dists = [(route_distance(r), idx) for idx, r in enumerate(route_list)]
        dists.sort(key=lambda x: (x[0], x[1]))
        idx1 = dists[0][1]
        idx2 = dists[1][1]
        r1 = route_list[idx1]
        r2 = route_list[idx2]
        merged = r1[:-1] + r2[1:]
        new_route_list = [r for i, r in enumerate(route_list) if i not in (idx1, idx2)]
        new_route_list.append(merged)
        route_list = new_route_list

    report_best_vrp(route_list)

    # Improvement loop with adaptive patience
    max_cycles = 5  # maximum full cycles of VND
    patience = 3  # cycles without improvement before stopping
    no_improve = 0
    for cycle in range(max_cycles):
        improved = False
        # Variable neighborhood descent: relocate, swap, then 2-opt on all routes
        # First relocate from longest route
        dists = [route_distance(r) for r in route_list]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = route_list[max_idx][1:-1]
        if interior:
            for cust in interior:
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = route_list[other_idx]
                    best_pos = None
                    best_delta = float('inf')
                    for pos in range(1, len(other_route)):
                        prev = other_route[pos-1]
                        nxt = other_route[pos] if pos < len(other_route) else 0
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if delta < best_delta - 1e-12:
                            best_delta = delta
                            best_pos = pos
                    new_routes = [list(r) for r in route_list]
                    new_routes[max_idx].remove(cust)
                    new_routes[other_idx].insert(best_pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_max - 1e-12:
                        route_list = new_routes
                        report_best_vrp(route_list)
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            # Swap between longest and another route
            dists = [route_distance(r) for r in route_list]
            max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
            interior = route_list[max_idx][1:-1]
            if interior:
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_interior = route_list[other_idx][1:-1]
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
                            if new_max < best_max - 1e-12:
                                route_list = new_routes
                                report_best_vrp(route_list)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        if not improved:
            # 2-opt on each route
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
                        if new_dist < best_dist - 1e-12:
                            best_dist = new_dist
                            best_route = new_route
                            found = True
                            break
                    if found:
                        break
                if found:
                    route_list[idx] = best_route
                    new_max = max(route_distance(r) for r in route_list)
                    if new_max < best_max - 1e-12:
                        report_best_vrp(route_list)
                    improved = True
                    break
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_routes if best_routes is not None else route_list
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

    # ---- Clarke-Wright with adaptive balance ----
    route_list = [[0, c, 0] for c in customers]
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((-s, i, j))
    heapq.heapify(savings)

    cust_to_route = {}
    route_endpoints = []
    route_dists = []
    for idx, route in enumerate(route_list):
        if len(route) == 3:
            cust = route[1]
            cust_to_route[cust] = idx
            route_endpoints.append((cust, cust, idx))
            route_dists.append(route_distance(route))

    total_dist = sum(route_dists)
    threshold_factor = 1.5

    while len(route_list) > truck_count and savings:
        avg_dist = total_dist / len(route_list)
        threshold = avg_dist * threshold_factor
        # Try to pop a saving that satisfies threshold
        found = False
        temp_heap = []
        while savings and not found:
            neg_s, i, j = heapq.heappop(savings)
            if i not in cust_to_route or j not in cust_to_route:
                continue
            ri = cust_to_route[i]
            rj = cust_to_route[j]
            if ri == rj:
                continue
            first_i, last_i, _ = route_endpoints[ri]
            first_j, last_j, _ = route_endpoints[rj]
            # Check merging possibilities
            possible = False
            if i == last_i and j == first_j:
                merge_dist = route_dists[ri] + route_dists[rj] - distance_matrix[0, last_i] - distance_matrix[0, first_j] + distance_matrix[last_i, first_j]
                possible = True
            elif j == last_j and i == first_i:
                merge_dist = route_dists[ri] + route_dists[rj] - distance_matrix[0, first_i] - distance_matrix[0, last_j] + distance_matrix[first_i, last_j]
                possible = True
            elif i == first_i and j == last_j:
                merge_dist = route_dists[ri] + route_dists[rj] - distance_matrix[0, first_i] - distance_matrix[0, last_j] + distance_matrix[first_i, last_j]
                possible = True
            elif j == first_j and i == last_i:
                merge_dist = route_dists[ri] + route_dists[rj] - distance_matrix[0, first_i] - distance_matrix[0, last_j] + distance_matrix[first_i, last_j]
                possible = True
            if possible and merge_dist <= threshold:
                # Perform merge
                if i == last_i and j == first_j:
                    new_route = route_list[ri][:-1] + route_list[rj][1:]
                elif j == last_j and i == first_i:
                    new_route = route_list[rj][:-1] + route_list[ri][1:]
                elif i == first_i and j == last_j:
                    new_route = route_list[rj][:-1] + route_list[ri][1:]
                else:
                    new_route = route_list[ri][:-1] + route_list[rj][1:]
                # Remove old routes and add new
                new_route_list = []
                new_dists = []
                new_endpoints = []
                for idx, r in enumerate(route_list):
                    if idx == ri or idx == rj:
                        continue
                    new_route_list.append(r)
                    new_dists.append(route_dists[idx])
                    new_endpoints.append(route_endpoints[idx])
                new_route_list.append(new_route)
                new_dists.append(route_distance(new_route))
                interior = new_route[1:-1]
                first_cust = interior[0] if interior else None
                last_cust = interior[-1] if interior else None
                new_endpoints.append((first_cust, last_cust, len(new_route_list)-1))
                route_list = new_route_list
                route_dists = new_dists
                route_endpoints = new_endpoints
                total_dist = sum(route_dists)
                # Update cust_to_route
                cust_to_route.clear()
                for idx2, r in enumerate(route_list):
                    for c in r[1:-1]:
                        cust_to_route[c] = idx2
                found = True
            else:
                # Save for later if not used
                heapq.heappush(temp_heap, (neg_s, i, j))
        if not found:
            # No merge satisfies threshold; break and fall back to merging without threshold
            # First push back the saved ones? Actually we need to restore heap
            # Simpler: break and then do standard merging below
            # Also push back the ones we popped but didn't use
            while temp_heap:
                heapq.heappush(savings, heapq.heappop(temp_heap))
            break
        else:
            # Merge performed, merge the remaining temp heap back
            while temp_heap:
                heapq.heappush(savings, heapq.heappop(temp_heap))

    # Fallback: if still more routes than truck_count, merge greedily (no threshold)
    while len(route_list) > truck_count and savings:
        neg_s, i, j = heapq.heappop(savings)
        if i not in cust_to_route or j not in cust_to_route:
            continue
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        if ri == rj:
            continue
        first_i, last_i, _ = route_endpoints[ri]
        first_j, last_j, _ = route_endpoints[rj]
        if i == last_i and j == first_j:
            new_route = route_list[ri][:-1] + route_list[rj][1:]
        elif j == last_j and i == first_i:
            new_route = route_list[rj][:-1] + route_list[ri][1:]
        elif i == first_i and j == last_j:
            new_route = route_list[rj][:-1] + route_list[ri][1:]
        elif j == first_j and i == last_i:
            new_route = route_list[ri][:-1] + route_list[rj][1:]
        else:
            continue
        new_route_list = []
        new_dists = []
        new_endpoints = []
        for idx, r in enumerate(route_list):
            if idx == ri or idx == rj:
                continue
            new_route_list.append(r)
            new_dists.append(route_dists[idx])
            new_endpoints.append(route_endpoints[idx])
        new_route_list.append(new_route)
        new_dists.append(route_distance(new_route))
        interior = new_route[1:-1]
        first_cust = interior[0] if interior else None
        last_cust = interior[-1] if interior else None
        new_endpoints.append((first_cust, last_cust, len(new_route_list)-1))
        route_list = new_route_list
        route_dists = new_dists
        route_endpoints = new_endpoints
        total_dist = sum(route_dists)
        cust_to_route.clear()
        for idx2, r in enumerate(route_list):
            for c in r[1:-1]:
                cust_to_route[c] = idx2

    # If still too many routes, merge smallest routes arbitrarily
    while len(route_list) > truck_count:
        dists = [(route_dists[i], i) for i in range(len(route_list))]
        dists.sort(key=lambda x: x[0])
        idx1 = dists[0][1]
        idx2 = dists[1][1]
        r1 = route_list[idx1]
        r2 = route_list[idx2]
        new_route = r1[:-1] + r2[1:]
        new_route_list = []
        new_dists = []
        new_endpoints = []
        for i, r in enumerate(route_list):
            if i == idx1 or i == idx2:
                continue
            new_route_list.append(r)
            new_dists.append(route_dists[i])
            new_endpoints.append(route_endpoints[i])
        new_route_list.append(new_route)
        new_dists.append(route_distance(new_route))
        interior = new_route[1:-1]
        first_cust = interior[0] if interior else None
        last_cust = interior[-1] if interior else None
        new_endpoints.append((first_cust, last_cust, len(new_route_list)-1))
        route_list = new_route_list
        route_dists = new_dists
        route_endpoints = new_endpoints
        total_dist = sum(route_dists)
        cust_to_route.clear()
        for idx2, r in enumerate(route_list):
            for c in r[1:-1]:
                cust_to_route[c] = idx2

    report_best_vrp(route_list)

    # ---- Improvement (unchanged from parent) ----
    max_iter = len(customers) * truck_count * 2
    for _ in range(max_iter):
        improved = False
        dists = [route_distance(r) for r in route_list]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        max_route = route_list[max_idx]
        interior = max_route[1:-1]
        if not interior:
            break
        for cust in interior:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = route_list[other_idx]
                best_pos = None
                best_delta = float('inf')
                for pos in range(1, len(other_route)):
                    prev = other_route[pos-1]
                    nxt = other_route[pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if delta < best_delta:
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
        if improved:
            continue
        # Swap
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_route = route_list[other_idx]
            interior_other = other_route[1:-1]
            if not interior_other:
                continue
            for cust_max in interior:
                for cust_other in interior_other:
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
        if improved:
            continue
        # 2-opt
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
        if not improved:
            break

    final_routes = best_routes if best_routes is not None else route_list
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes
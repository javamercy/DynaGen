import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        for _ in range(truck_count - m):
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # --- Greedy insertion construction ---
    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    # Sort customers by distance to depot descending
    depot_dist = [distance_matrix[0, c] for c in customers]
    sorted_cust = sorted(customers, key=lambda c: (-depot_dist[c-1], c))
    
    def route_dist(route):
        total = 0
        for a in range(len(route)-1):
            total += distance_matrix[route[a], route[a+1]]
        return total
    
    for cust in sorted_cust:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            # Evaluate insertion at each position (after depot 0, before depot 0)
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_dist(new_route)
                # Compute new max across all routes
                new_max = new_dist
                for j, other in enumerate(routes):
                    if j != r_idx:
                        d = route_dist(other)
                        if d > new_max:
                            new_max = d
                if new_max < best_max or (new_max == best_max and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        # Insert
        routes[best_route_idx].insert(best_pos, cust)
    
    # Compute initial best
    def compute_max():
        maxd = 0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd
    
    best_max = compute_max()
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)
    
    # --- Improvement: round-robin best-improvement (same as parent) ---
    max_passes = n * n
    for _ in range(max_passes):
        dists = [route_dist(r) for r in routes]
        current_max = max(dists)
        improved = False

        # Order routes by distance descending, then index ascending
        order = sorted(range(len(routes)), key=lambda i: (-dists[i], i))

        # Best 2-opt on each route
        for idx in order:
            route = routes[idx]
            if len(route) <= 3:
                continue
            best_local = route_dist(route)
            best_route = route[:]
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_local:
                        best_local = new_dist
                        best_route = new_route
            if best_local < route_dist(route):
                routes[idx] = best_route
                new_max = compute_max()
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                improved = True
                break

        if improved:
            continue

        dists = [route_dist(r) for r in routes]
        current_max = max(dists)
        order = sorted(range(len(routes)), key=lambda i: (-dists[i], i))
        longest_idx = order[0]

        # Best relocate from longest route
        best_improvement = 0.0
        best_move = None
        src_route = routes[longest_idx]
        for cust_pos in range(1, len(src_route) - 1):
            cust = src_route[cust_pos]
            new_src = src_route[:cust_pos] + src_route[cust_pos+1:]
            dist_src = route_dist(new_src)
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos in range(1, len(dst_route)):
                    new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                    dist_dst = route_dist(new_dst)
                    other_dists = [route_dist(r) for i, r in enumerate(routes) if i not in (longest_idx, dst_idx)]
                    new_max = max([dist_src, dist_dst] + other_dists)
                    improvement = current_max - new_max
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (cust, cust_pos, dst_idx, pos)
                    elif improvement == best_improvement and best_move is not None:
                        (ocust, ocust_pos, odst_idx, opos) = best_move
                        if cust < ocust or (cust == ocust and cust_pos < ocust_pos) or (cust == ocust and cust_pos == ocust_pos and dst_idx < odst_idx) or (cust == ocust and cust_pos == ocust_pos and dst_idx == odst_idx and pos < opos):
                            best_improvement = improvement
                            best_move = (cust, cust_pos, dst_idx, pos)

        if best_move and best_improvement > 0:
            cust, cust_pos, dst_idx, pos = best_move
            routes[longest_idx].pop(cust_pos)
            routes[dst_idx].insert(pos, cust)
            new_max = compute_max()
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            improved = True

        if improved:
            continue

        # Best swap from longest route
        best_improvement = 0.0
        best_move = None
        src_idx = longest_idx
        src_route = routes[src_idx]
        for pos_i in range(1, len(src_route) - 1):
            cust_i = src_route[pos_i]
            for dst_idx in range(len(routes)):
                if dst_idx == src_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos_j in range(1, len(dst_route) - 1):
                    cust_j = dst_route[pos_j]
                    new_src = src_route[:pos_i] + [cust_j] + src_route[pos_i+1:]
                    new_dst = dst_route[:pos_j] + [cust_i] + dst_route[pos_j+1:]
                    new_dist_src = route_dist(new_src)
                    new_dist_dst = route_dist(new_dst)
                    other_dists = [route_dist(r) for i, r in enumerate(routes) if i not in (src_idx, dst_idx)]
                    new_max = max([new_dist_src, new_dist_dst] + other_dists)
                    improvement = current_max - new_max
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (cust_i, pos_i, cust_j, pos_j, dst_idx)
                    elif improvement == best_improvement and best_move is not None:
                        (ocust_i, opos_i, ocust_j, opos_j, odst_idx) = best_move
                        if (cust_i < ocust_i or (cust_i == ocust_i and pos_i < opos_i) or
                            (cust_i == ocust_i and pos_i == opos_i and cust_j < ocust_j) or
                            (cust_i == ocust_i and pos_i == opos_i and cust_j == ocust_j and dst_idx < odst_idx)):
                            best_improvement = improvement
                            best_move = (cust_i, pos_i, cust_j, pos_j, dst_idx)

        if best_move and best_improvement > 0:
            cust_i, pos_i, cust_j, pos_j, dst_idx = best_move
            routes[src_idx][pos_i] = cust_j
            routes[dst_idx][pos_j] = cust_i
            new_max = compute_max()
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            improved = True

        if not improved:
            break

    return best_routes
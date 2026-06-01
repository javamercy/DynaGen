import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in range(1, n)]
        for _ in range(truck_count - m):
            routes.append([0, 0])
        return routes

    # --- Greedy insertion construction ---
    routes = [[0, 0] for _ in range(truck_count)]
    # Order customers by distance from depot descending
    depot_dists = [distance_matrix[0, i] for i in range(1, n)]
    order = sorted(range(1, n), key=lambda i: -depot_dists[i-1])
    
    def route_dist(route):
        total = 0
        for a in range(len(route)-1):
            total += distance_matrix[route[a], route[a+1]]
        return total

    for cust in order:
        best_max = math.inf
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            if len(route) == 2:
                # Empty route: just insert customer
                new_route = [0, cust, 0]
                new_max = route_dist(new_route)
                if new_max < best_max or (new_max == best_max and r_idx < best_route_idx):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = 1
            else:
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_dist(new_route)
                    # Compute max after insertion (others unchanged)
                    other_dists = [route_dist(r) for i, r in enumerate(routes) if i != r_idx]
                    cand_max = max([new_dist] + other_dists)
                    if cand_max < best_max or (cand_max == best_max and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                        best_max = cand_max
                        best_route_idx = r_idx
                        best_pos = pos
        routes[best_route_idx].insert(best_pos, cust)

    # Report initial best
    def compute_max():
        return max(route_dist(r) for r in routes)
    best_max = compute_max()
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    # --- Improvement ---
    max_passes = n * truck_count * 2  # finite bound
    for _ in range(max_passes):
        improved = False
        current_max = compute_max()
        
        # 2-opt: try all intra-route moves, apply best one that reduces max
        best_2opt = None  # (route_idx, i, j, new_max)
        best_2opt_improvement = 0
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    old_dist = route_dist(route)
                    # compute new max
                    other_dists = [route_dist(r) for k, r in enumerate(routes) if k != r_idx]
                    cand_max = max([new_dist] + other_dists)
                    improvement = current_max - cand_max
                    if improvement > best_2opt_improvement or (improvement == best_2opt_improvement and best_2opt is not None and (r_idx < best_2opt[0] or (r_idx == best_2opt[0] and i < best_2opt[1]) or (r_idx == best_2opt[0] and i == best_2opt[1] and j < best_2opt[2]))):
                        best_2opt_improvement = improvement
                        best_2opt = (r_idx, i, j, cand_max)
        if best_2opt and best_2opt_improvement > 0:
            r_idx, i, j, new_max = best_2opt
            route = routes[r_idx]
            routes[r_idx] = route[:i] + route[i:j+1][::-1] + route[j+1:]
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            improved = True
            continue  # restart pass after an improvement

        # Relocate from the longest route: try moving each customer to any other route, best improvement
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(dists)), key=lambda i: (dists[i], -i))  # tie: smaller index
        best_reloc = None  # (cust_pos, dst_idx, dst_pos, new_max)
        best_reloc_improvement = 0
        src_route = routes[longest_idx]
        if len(src_route) > 2:
            for cust_pos in range(1, len(src_route)-1):
                cust = src_route[cust_pos]
                new_src = src_route[:cust_pos] + src_route[cust_pos+1:]
                new_src_dist = route_dist(new_src)
                for dst_idx in range(truck_count):
                    if dst_idx == longest_idx:
                        continue
                    dst_route = routes[dst_idx]
                    for pos in range(1, len(dst_route)):
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        new_dst_dist = route_dist(new_dst)
                        other_dists = [route_dist(r) for k, r in enumerate(routes) if k not in (longest_idx, dst_idx)]
                        cand_max = max([new_src_dist, new_dst_dist] + other_dists)
                        improvement = current_max - cand_max
                        # tie-breaking: smaller cust, then smaller cust_pos, then smaller dst_idx, then pos
                        if improvement > best_reloc_improvement or (improvement == best_reloc_improvement and best_reloc is not None and (
                            cust < best_reloc[0] or (cust == best_reloc[0] and cust_pos < best_reloc[1]) or 
                            (cust == best_reloc[0] and cust_pos == best_reloc[1] and dst_idx < best_reloc[2]) or
                            (cust == best_reloc[0] and cust_pos == best_reloc[1] and dst_idx == best_reloc[2] and pos < best_reloc[3]))):
                            best_reloc_improvement = improvement
                            best_reloc = (cust, cust_pos, dst_idx, pos, cand_max)
        if best_reloc and best_reloc_improvement > 0:
            cust, cust_pos, dst_idx, pos, new_max = best_reloc
            routes[longest_idx].pop(cust_pos)
            routes[dst_idx].insert(pos, cust)
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            improved = True

        if not improved:
            break

    return best_routes
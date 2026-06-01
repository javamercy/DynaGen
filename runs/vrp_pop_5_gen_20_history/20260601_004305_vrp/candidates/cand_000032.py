import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # Step 1: Savings construction (as in cand_000009)
    routes = [[0, i, 0] for i in range(1, n)]
    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_savings(routes):
        savings = []
        for i, r_i in enumerate(routes):
            if len(r_i) == 2:
                continue
            last_i = r_i[-2]
            first_i = r_i[1]
            for j, r_j in enumerate(routes):
                if i == j or len(r_j) == 2:
                    continue
                first_j = r_j[1]
                last_j = r_j[-2]
                s1 = distance_matrix[0, last_i] + distance_matrix[0, first_j] - distance_matrix[last_i, first_j]
                savings.append((s1, i, j, 0))
                s2 = distance_matrix[0, last_j] + distance_matrix[0, first_i] - distance_matrix[last_j, first_i]
                savings.append((s2, i, j, 1))
        savings.sort(reverse=True, key=lambda x: x[0])
        return savings

    def merge_routes(routes, i, j, mtype):
        if mtype == 0:
            new_route = routes[i][:-1] + routes[j][1:]
        else:
            new_route = routes[j][:-1] + routes[i][1:]
        if i > j:
            del routes[i]
            del routes[j]
        else:
            del routes[j]
            del routes[i]
        routes.append(new_route)
        return routes

    while len(routes) > truck_count:
        savings = compute_savings(routes)
        if not savings:
            break
        for saving, i, j, mtype in savings:
            if i < len(routes) and j < len(routes):
                routes = merge_routes(routes, i, j, mtype)
                break

    best_max = max(route_dist(r) for r in routes)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    # Step 2: Round-robin best-improvement (adapted from cand_000027)
    max_passes = n * n
    for _ in range(max_passes):
        dists = [route_dist(r) for r in routes]
        current_max = max(dists)
        improved = False

        # Sort routes by distance descending, then index ascending
        order = sorted(range(len(routes)), key=lambda i: (-dists[i], i))

        # Best 2-opt on each route
        for idx in order:
            route = routes[idx]
            if len(route) <= 3:
                continue
            best_local = route_dist(route)
            best_route = route[:]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_local:
                        best_local = new_dist
                        best_route = new_route
            if best_local < route_dist(route):
                routes[idx] = best_route
                new_max = max(route_dist(r) for r in routes)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                improved = True
                break

        if improved:
            continue

        # Recompute order after potential changes (none if not improved)
        dists = [route_dist(r) for r in routes]
        current_max = max(dists)
        order = sorted(range(len(routes)), key=lambda i: (-dists[i], i))
        longest_idx = order[0]

        # Best relocate from longest route
        best_improvement = 0.0
        best_move = None
        src_route = routes[longest_idx]
        for cust_pos in range(1, len(src_route)-1):
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
                        ocust, ocust_pos, odst_idx, opos = best_move
                        if (cust < ocust or 
                            (cust == ocust and cust_pos < ocust_pos) or
                            (cust == ocust and cust_pos == ocust_pos and dst_idx < odst_idx) or
                            (cust == ocust and cust_pos == ocust_pos and dst_idx == odst_idx and pos < opos)):
                            best_improvement = improvement
                            best_move = (cust, cust_pos, dst_idx, pos)

        if best_move and best_improvement > 0:
            cust, cust_pos, dst_idx, pos = best_move
            routes[longest_idx].pop(cust_pos)
            routes[dst_idx].insert(pos, cust)
            new_max = max(route_dist(r) for r in routes)
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
        for pos_i in range(1, len(src_route)-1):
            cust_i = src_route[pos_i]
            for dst_idx in range(len(routes)):
                if dst_idx == src_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos_j in range(1, len(dst_route)-1):
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
                        ocust_i, opos_i, ocust_j, opos_j, odst_idx = best_move
                        if (cust_i < ocust_i or
                            (cust_i == ocust_i and pos_i < opos_i) or
                            (cust_i == ocust_i and pos_i == opos_i and cust_j < ocust_j) or
                            (cust_i == ocust_i and pos_i == opos_i and cust_j == ocust_j and dst_idx < odst_idx)):
                            best_improvement = improvement
                            best_move = (cust_i, pos_i, cust_j, pos_j, dst_idx)

        if best_move and best_improvement > 0:
            cust_i, pos_i, cust_j, pos_j, dst_idx = best_move
            routes[src_idx][pos_i] = cust_j
            routes[dst_idx][pos_j] = cust_i
            new_max = max(route_dist(r) for r in routes)
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            improved = True

        if not improved:
            break

    return best_routes
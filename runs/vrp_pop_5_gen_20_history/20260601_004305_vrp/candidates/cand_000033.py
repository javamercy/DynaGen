import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # initial routes: each customer alone
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

    # merge until truck_count
    while len(routes) > truck_count:
        savings = compute_savings(routes)
        if not savings:
            break
        for saving, i, j, mtype in savings:
            if i >= len(routes) or j >= len(routes):
                continue
            routes = merge_routes(routes, i, j, mtype)
            break

    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in routes)
    report_best_vrp(best_routes)

    # improvement phase: simple pass until no improvement
    max_passes = n * n
    for _ in range(max_passes):
        dists = [route_dist(r) for r in routes]
        current_max = max(dists)
        improved = False

        # Or-opt on all routes (longest first)
        order = sorted(range(len(routes)), key=lambda i: (-dists[i], i))
        for idx in order:
            route = routes[idx]
            if len(route) <= 3:
                continue
            best_local = route_dist(route)
            best_route = route[:]
            # try removing segment [i:j] (1 to 3 customers) and insert at various positions
            for seg_len in range(1, min(4, len(route)-2)):
                for i in range(1, len(route)-seg_len-1):
                    j = i + seg_len - 1
                    segment = route[i:j+1]
                    remaining = route[:i] + route[j+1:]
                    # try inserting segment at each position in remaining
                    for pos in range(1, len(remaining)):
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        if new_route[0] != 0 or new_route[-1] != 0:
                            continue
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
                break  # restart Or-opt after improvement

        if improved:
            continue

        # inter-route relocation
        for src_idx in range(len(routes)):
            src_route = routes[src_idx]
            for cust_idx in range(1, len(src_route)-1):
                cust = src_route[cust_idx]
                new_src = src_route[:cust_idx] + src_route[cust_idx+1:]
                if new_src[0] != 0 or new_src[-1] != 0:
                    continue
                best_new_max = None
                best_dst_idx = None
                best_pos = None
                for dst_idx in range(len(routes)):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    for pos in range(1, len(dst_route)):
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        if new_dst[0] != 0 or new_dst[-1] != 0:
                            continue
                        new_dist_src = route_dist(new_src)
                        new_dist_dst = route_dist(new_dst)
                        other_dists = [route_dist(r) for i2, r in enumerate(routes) if i2 not in (src_idx, dst_idx)]
                        candidate_max = max([new_dist_src, new_dist_dst] + other_dists)
                        if candidate_max < best_max:
                            if best_new_max is None or candidate_max < best_new_max:
                                best_new_max = candidate_max
                                best_dst_idx = dst_idx
                                best_pos = pos
                if best_new_max is not None:
                    routes[src_idx] = new_src
                    routes[best_dst_idx] = routes[best_dst_idx][:best_pos] + [cust] + routes[best_dst_idx][best_pos:]
                    best_max = best_new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best_routes
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

    # VND improvement phase
    max_iter = n
    for _ in range(max_iter):
        improved = False
        dists = [route_dist(r) for r in routes]

        # Intra-route 2-opt (best improvement)
        order = sorted(range(len(routes)), key=lambda i: (-dists[i], i))
        best_2opt = None
        best_2opt_val = None
        for idx in order:
            route = routes[idx]
            if len(route) <= 3:
                continue
            best_local_route = None
            best_local_dist = route_dist(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if new_route[0] != 0 or new_route[-1] != 0:
                        continue
                    new_dist = route_dist(new_route)
                    if new_dist < best_local_dist:
                        best_local_dist = new_dist
                        best_local_route = new_route
            if best_local_route is not None:
                candidate_max = max(route_dist(r) if k != idx else best_local_dist for k, r in enumerate(routes))
                if candidate_max < best_max and (best_2opt is None or candidate_max < best_2opt_val):
                    best_2opt = (idx, best_local_route)
                    best_2opt_val = candidate_max
        if best_2opt is not None:
            idx, new_route = best_2opt
            routes[idx] = new_route
            best_max = best_2opt_val
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            improved = True
            continue

        # Inter-route relocation (best improvement)
        best_reloc = None
        best_reloc_val = None
        for src_idx in range(len(routes)):
            src_route = routes[src_idx]
            for cust_pos in range(1, len(src_route)-1):
                cust = src_route[cust_pos]
                new_src = src_route[:cust_pos] + src_route[cust_pos+1:]
                if new_src[0] != 0 or new_src[-1] != 0:
                    continue
                for dst_idx in range(len(routes)):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    for pos in range(1, len(dst_route)):
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        if new_dst[0] != 0 or new_dst[-1] != 0:
                            continue
                        new_src_dist = route_dist(new_src)
                        new_dst_dist = route_dist(new_dst)
                        other_dists = [route_dist(r) for i2, r in enumerate(routes) if i2 not in (src_idx, dst_idx)]
                        candidate_max = max([new_src_dist, new_dst_dist] + other_dists)
                        if candidate_max < best_max and (best_reloc is None or candidate_max < best_reloc_val):
                            best_reloc = (src_idx, cust_pos, dst_idx, pos, new_src, new_dst)
                            best_reloc_val = candidate_max
        if best_reloc is not None:
            src_idx, cust_pos, dst_idx, pos, new_src, new_dst = best_reloc
            routes[src_idx] = new_src
            routes[dst_idx] = new_dst
            best_max = best_reloc_val
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            improved = True
            continue

        # Inter-route 2-opt* (best improvement)
        best_2optstar = None
        best_2optstar_val = None
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 2 or len(route_j) <= 2:
                    continue
                for u in range(1, len(route_i)-1):
                    for v in range(1, len(route_j)-1):
                        new_i = route_i[:u] + route_j[v:]
                        new_j = route_j[:v] + route_i[u:]
                        if new_i[0] != 0 or new_i[-1] != 0 or new_j[0] != 0 or new_j[-1] != 0:
                            continue
                        new_i_dist = route_dist(new_i)
                        new_j_dist = route_dist(new_j)
                        other_dists = [route_dist(r) for k, r in enumerate(routes) if k not in (i, j)]
                        candidate_max = max([new_i_dist, new_j_dist] + other_dists)
                        if candidate_max < best_max and (best_2optstar is None or candidate_max < best_2optstar_val):
                            best_2optstar = (i, j, u, v, new_i, new_j)
                            best_2optstar_val = candidate_max
        if best_2optstar is not None:
            i, j, u, v, new_i, new_j = best_2optstar
            routes[i] = new_i
            routes[j] = new_j
            best_max = best_2optstar_val
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            improved = True
            continue

        if not improved:
            break

    return best_routes
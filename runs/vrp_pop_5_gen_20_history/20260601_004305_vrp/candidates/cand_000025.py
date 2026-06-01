import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    routes = [[0, i, 0] for i in range(1, n)]
    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_dist(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

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
        savings.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
        return savings

    while len(routes) > truck_count:
        savings = compute_savings(routes)
        if not savings:
            break
        merged = False
        for saving, i, j, mtype in savings:
            if i >= len(routes) or j >= len(routes) or i == j:
                continue
            if len(routes[i]) == 2 or len(routes[j]) == 2:
                continue
            if mtype == 0:
                new_route = routes[i][:-1] + routes[j][1:]
            else:
                new_route = routes[j][:-1] + routes[i][1:]
            if new_route[0] != 0 or new_route[-1] != 0:
                continue
            if i > j:
                del routes[i]
                del routes[j]
            else:
                del routes[j]
                del routes[i]
            routes.append(new_route)
            merged = True
            break
        if not merged:
            idxs = [i for i, r in enumerate(routes) if len(r) > 2]
            if len(idxs) >= 2:
                i, j = idxs[0], idxs[1]
                new_route = routes[i][:-1] + routes[j][1:]
                if i > j:
                    del routes[i]
                    del routes[j]
                else:
                    del routes[j]
                    del routes[i]
                routes.append(new_route)
            else:
                break

    while len(routes) < truck_count:
        routes.append([0, 0])

    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in routes)
    report_best_vrp(best_routes)

    max_passes = n * 2
    for _ in range(max_passes):
        dists = [route_dist(r) for r in routes]
        max_idx = max(range(len(routes)), key=lambda i: dists[i])
        max_dist = dists[max_idx]
        improved = False

        route = routes[max_idx]
        if len(route) > 3:
            best_local = max_dist
            best_route = route[:]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if new_route[0] != 0 or new_route[-1] != 0:
                        continue
                    new_dist = route_dist(new_route)
                    if new_dist < best_local:
                        best_local = new_dist
                        best_route = new_route
            if best_local < max_dist:
                routes[max_idx] = best_route
                new_max = max(route_dist(r) for r in routes)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    improved = True

        if not improved:
            route = routes[max_idx]
            for cust_idx in range(1, len(route)-1):
                cust = route[cust_idx]
                new_long = route[:cust_idx] + route[cust_idx+1:]
                dist_long = route_dist(new_long)
                for other_idx in range(len(routes)):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        new_other = other_route[:pos] + [cust] + other_route[pos:]
                        if new_other[0] != 0 or new_other[-1] != 0:
                            continue
                        new_dist_other = route_dist(new_other)
                        other_dists = [route_dist(r) for idx2, r in enumerate(routes) if idx2 not in (max_idx, other_idx)]
                        new_max_candidate = max(dist_long, new_dist_other, *other_dists)
                        if new_max_candidate < best_max:
                            routes[max_idx] = new_long
                            routes[other_idx] = new_other
                            best_max = new_max_candidate
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        if not improved:
            break

    return best_routes
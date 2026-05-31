import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = set(range(1, n))

    def route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    # Regret insertion
    while unvisited:
        best_cust = None
        best_route = -1
        best_pos = -1
        best_regret = -float('inf')
        best_new_dist = float('inf')
        for cust in unvisited:
            # compute best and second best insertion costs
            costs = []
            for r_idx, route in enumerate(routes):
                cur_dist = route_distance(route)
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_distance(new_route)
                    increase = new_dist - cur_dist
                    costs.append((increase, r_idx, pos, new_dist))
            if not costs:
                continue
            costs.sort(key=lambda x: (x[0], x[1], x[2]))
            best = costs[0]
            second_best = costs[1] if len(costs) > 1 else (best[0], best[1], best[2], best[3])  # fallback
            regret = second_best[0] - best[0]
            # tie-breaking on regret, then by smaller new_dist, then route index
            if regret > best_regret or (regret == best_regret and (best[3] < best_new_dist or (abs(best[3]-best_new_dist) < 1e-9 and best[1] < best_route))):
                best_regret = regret
                best_cust = cust
                best_route = best[1]
                best_pos = best[2]
                best_new_dist = best[3]
        # insert customer
        route = routes[best_route]
        routes[best_route] = route[:best_pos] + [best_cust] + route[best_pos:]
        unvisited.remove(best_cust)

    # report initial solution
    report_best_vrp(routes)

    # improvement: 2-opt within each route
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        max_iter = n
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            best_i = -1
            best_j = -1
            best_dist = route_distance(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-9:
                        best_dist = new_dist
                        best_i, best_j = i, j
                        improved = True
            if improved:
                route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                routes[r_idx] = route

    # relocate between routes to reduce max distance
    max_iter = n * truck_count
    improved = True
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        current_max = max(route_distance(r) for r in routes)
        for cust in range(1, n):
            src_route_idx = None
            cust_pos_in_src = None
            for idx, route in enumerate(routes):
                if cust in route:
                    src_route_idx = idx
                    cust_pos_in_src = route.index(cust)
                    break
            if src_route_idx is None:
                continue
            src_route = routes[src_route_idx]
            new_src = src_route[:cust_pos_in_src] + src_route[cust_pos_in_src+1:]
            src_dist = route_distance(new_src)
            for dst_route_idx in range(truck_count):
                if dst_route_idx == src_route_idx:
                    continue
                dst_route = routes[dst_route_idx]
                for pos in range(1, len(dst_route)):
                    new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                    dst_dist = route_distance(new_dst)
                    new_max = max(src_dist, dst_dist)
                    for other_idx in range(truck_count):
                        if other_idx != src_route_idx and other_idx != dst_route_idx:
                            new_max = max(new_max, route_distance(routes[other_idx]))
                    if new_max < current_max - 1e-9:
                        routes[src_route_idx] = new_src
                        routes[dst_route_idx] = new_dst
                        current_max = new_max
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            report_best_vrp(routes)

    return routes
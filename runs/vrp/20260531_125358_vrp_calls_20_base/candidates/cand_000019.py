import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = list(range(1, n))
    # sort customers by distance from depot descending
    unvisited.sort(key=lambda c: distance_matrix[0, c], reverse=True)

    def route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    # minimax insertion construction
    current_distances = [0.0 for _ in range(truck_count)]  # distance of each route
    for cust in unvisited:
        best_route = -1
        best_pos = -1
        best_new_max = float('inf')
        for r_idx, route in enumerate(routes):
            cur_dist = current_distances[r_idx]
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route)
                # compute new max over all routes
                new_max = new_dist
                for other_idx, other_dist in enumerate(current_distances):
                    if other_idx != r_idx:
                        if other_dist > new_max:
                            new_max = other_dist
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    best_route = r_idx
                    best_pos = pos
                elif abs(new_max - best_new_max) < 1e-12:
                    # tie-break: smaller route index, then smaller position
                    if r_idx < best_route or (r_idx == best_route and pos < best_pos):
                        best_new_max = new_max
                        best_route = r_idx
                        best_pos = pos
        # insert
        route = routes[best_route]
        routes[best_route] = route[:best_pos] + [cust] + route[best_pos:]
        current_distances[best_route] = route_distance(routes[best_route])

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
                current_distances[r_idx] = route_distance(route)

    # relocate between routes to reduce max distance
    improved = True
    max_iter = n * truck_count
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        current_max = max(current_distances)
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
                    new_max = max(src_dist, dst_dist, max(current_distances[:src_route_idx] + current_distances[src_route_idx+1:dst_route_idx] + current_distances[dst_route_idx+1:]))
                    if new_max < current_max - 1e-9:
                        routes[src_route_idx] = new_src
                        routes[dst_route_idx] = new_dst
                        current_distances[src_route_idx] = src_dist
                        current_distances[dst_route_idx] = dst_dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            report_best_vrp(routes)

    return routes
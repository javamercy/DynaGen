import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # initial empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = list(range(1, n))
    # sort by distance from depot descending
    unvisited.sort(key=lambda c: distance_matrix[0, c], reverse=True)

    def route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    # greedy insertion construction
    for cust in unvisited:
        best_route_idx = -1
        best_pos = -1
        best_new_dist = float('inf')
        for r_idx, route in enumerate(routes):
            cur_dist = route_distance(route)
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route)
                # minimize new distance, break ties by route index
                if new_dist < best_new_dist - 1e-12:
                    best_new_dist = new_dist
                    best_route_idx = r_idx
                    best_pos = pos
                elif abs(new_dist - best_new_dist) < 1e-12 and r_idx < best_route_idx:
                    best_route_idx = r_idx
                    best_pos = pos
        # insert
        route = routes[best_route_idx]
        routes[best_route_idx] = route[:best_pos] + [cust] + route[best_pos:]

    report_best_vrp(routes)

    # 2-opt for each route (adaptive iterations)
    def two_opt(route, max_iter=None):
        if max_iter is None:
            max_iter = max(10, 2 * (len(route) - 2))  # based on number of customers
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            best_improvement = 0.0
            best_i = best_j = -1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    cur_dist = route_distance(route)
                    new_dist = route_distance(new_route)
                    if new_dist < cur_dist - 1e-12:
                        improvement = cur_dist - new_dist
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_i, best_j = i, j
                            improved = True
            if improved:
                route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
        return route

    for r_idx in range(truck_count):
        if len(routes[r_idx]) > 2:
            routes[r_idx] = two_opt(routes[r_idx])

    report_best_vrp(routes)

    # adaptive balancing loop (move from max to min route)
    lengths = [route_distance(r) for r in routes]
    no_improve_count = 0
    max_outer_iter = n * truck_count
    for _ in range(max_outer_iter):
        # find longest and shortest route
        max_idx = max(range(truck_count), key=lambda i: lengths[i])
        min_idx = min(range(truck_count), key=lambda i: lengths[i])
        if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
            break
        max_route = routes[max_idx]
        min_route = routes[min_idx]
        best_cust = None
        best_remove_pos = -1
        best_insert_pos = -1
        best_new_max = float('inf')
        # evaluate moving each customer from max_route to min_route
        for pos_remove in range(1, len(max_route)-1):
            cust = max_route[pos_remove]
            new_max_route = max_route[:pos_remove] + max_route[pos_remove+1:]
            new_max_len = route_distance(new_max_route)
            for pos_insert in range(1, len(min_route)):
                new_min_route = min_route[:pos_insert] + [cust] + min_route[pos_insert:]
                new_min_len = route_distance(new_min_route)
                new_max = new_max_len
                if new_min_len > new_max:
                    new_max = new_min_len
                # consider other routes unchanged
                for other_idx in range(truck_count):
                    if other_idx == max_idx or other_idx == min_idx:
                        continue
                    if lengths[other_idx] > new_max:
                        new_max = lengths[other_idx]
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    best_cust = cust
                    best_remove_pos = pos_remove
                    best_insert_pos = pos_insert
        if best_cust is not None and best_new_max < max(lengths) - 1e-12:
            # perform move
            routes[max_idx] = routes[max_idx][:best_remove_pos] + routes[max_idx][best_remove_pos+1:]
            routes[min_idx] = routes[min_idx][:best_insert_pos] + [best_cust] + routes[min_idx][best_insert_pos:]
            lengths[max_idx] = route_distance(routes[max_idx])
            lengths[min_idx] = route_distance(routes[min_idx])
            no_improve_count = 0
            report_best_vrp(routes)
        else:
            no_improve_count += 1
            if no_improve_count >= 3:
                break

    # final 2-opt on all routes
    for r_idx in range(truck_count):
        if len(routes[r_idx]) > 2:
            routes[r_idx] = two_opt(routes[r_idx])
    report_best_vrp(routes)

    return routes
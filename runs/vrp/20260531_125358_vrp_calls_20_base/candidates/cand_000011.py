import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Construction: greedy insertion based on distance from depot
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = list(range(1, n))
    unvisited.sort(key=lambda c: distance_matrix[0, c], reverse=True)
    
    def route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist
    
    for cust in unvisited:
        best_route = -1
        best_pos = -1
        best_new_dist = float('inf')
        for r_idx, route in enumerate(routes):
            cur_dist = route_distance(route)
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route)
                if new_dist < best_new_dist or (new_dist == best_new_dist and r_idx < best_route):
                    best_new_dist = new_dist
                    best_route = r_idx
                    best_pos = pos
        route = routes[best_route]
        routes[best_route] = route[:best_pos] + [cust] + route[best_pos:]
    
    report_best_vrp(routes)
    
    # Improvement: relocate moves to reduce max distance
    max_iter = n * truck_count
    for _ in range(max_iter):
        current_max = max(route_distance(r) for r in routes)
        improved = False
        for cust in range(1, n):
            # Find current route and position of cust
            src_route_idx = None
            src_pos = None
            for r_idx, route in enumerate(routes):
                if cust in route:
                    src_route_idx = r_idx
                    src_pos = route.index(cust)
                    break
            if src_route_idx is None:
                continue
            src_route = routes[src_route_idx]
            # Remove cust from source
            new_src = src_route[:src_pos] + src_route[src_pos+1:]
            src_dist = route_distance(new_src)
            # Consider all destination routes and positions
            for dst_route_idx in range(truck_count):
                if dst_route_idx == src_route_idx:
                    # Intra-route: insert at different position
                    for pos in range(1, len(new_src)):
                        new_dst = new_src[:pos] + [cust] + new_src[pos:]
                        dst_dist = route_distance(new_dst)
                        # Compute new max
                        new_max = src_dist if src_dist > dst_dist else dst_dist
                        for other_idx in range(truck_count):
                            if other_idx != src_route_idx:
                                new_max = max(new_max, route_distance(routes[other_idx]))
                        if new_max < current_max - 1e-9:
                            routes[src_route_idx] = new_dst
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                else:
                    # Inter-route: insert into another route
                    dst_route = routes[dst_route_idx]
                    for pos in range(1, len(dst_route)):
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        dst_dist = route_distance(new_dst)
                        # Compute new max
                        new_max = src_dist if src_dist > dst_dist else dst_dist
                        for other_idx in range(truck_count):
                            if other_idx != src_route_idx and other_idx != dst_route_idx:
                                new_max = max(new_max, route_distance(routes[other_idx]))
                        if new_max < current_max - 1e-9:
                            routes[src_route_idx] = new_src
                            routes[dst_route_idx] = new_dst
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        if not improved:
            break
    
    return routes
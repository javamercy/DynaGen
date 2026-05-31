import numpy as np

def report_best_vrp(routes):
    pass  # placeholder, assumed to be provided externally

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
    
    # Improvement: steepest descent relocate + intra-route 2-opt
    max_iter = n * truck_count * 2
    for _ in range(max_iter):
        original_max = max(route_distance(r) for r in routes)
        original_total = sum(route_distance(r) for r in routes)
        best_move = None
        best_new_max = original_max
        best_new_total = original_total
        
        # Evaluate all relocate moves
        for cust in range(1, n):
            # Find current route and position
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
            new_src = src_route[:src_pos] + src_route[src_pos+1:]
            
            # Intra-route relocate
            for pos in range(1, len(new_src)):
                new_dst = new_src[:pos] + [cust] + new_src[pos:]
                new_routes = routes.copy()
                new_routes[src_route_idx] = new_dst
                new_max = max(route_distance(r) for r in new_routes)
                new_total = sum(route_distance(r) for r in new_routes)
                if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                    best_new_max = new_max
                    best_new_total = new_total
                    best_move = ('relocate_intra', cust, src_route_idx, src_pos, pos, new_dst)
            
            # Inter-route relocate
            for dst_route_idx in range(truck_count):
                if dst_route_idx == src_route_idx:
                    continue
                dst_route = routes[dst_route_idx]
                for pos in range(1, len(dst_route)):
                    new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                    new_routes = routes.copy()
                    new_routes[src_route_idx] = new_src
                    new_routes[dst_route_idx] = new_dst
                    new_max = max(route_distance(r) for r in new_routes)
                    new_total = sum(route_distance(r) for r in new_routes)
                    if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_move = ('relocate_inter', cust, src_route_idx, src_pos, dst_route_idx, pos, new_src, new_dst)
        
        # Evaluate all intra-route 2-opt moves on each route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_routes = routes.copy()
                    new_routes[r_idx] = new_route
                    new_max = max(route_distance(r) for r in new_routes)
                    new_total = sum(route_distance(r) for r in new_routes)
                    if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_move = ('2opt', r_idx, i, j, new_route)
        
        if best_move is None:
            break
        # Apply best move if it improves max or total
        if best_new_max < original_max or (best_new_max == original_max and best_new_total < original_total):
            if best_move[0] == 'relocate_intra':
                routes[best_move[2]] = best_move[5]
            elif best_move[0] == 'relocate_inter':
                routes[best_move[2]] = best_move[6]
                routes[best_move[4]] = best_move[7]
            else:  # 2opt
                routes[best_move[1]] = best_move[4]
            report_best_vrp(routes)
        else:
            break
    
    return routes
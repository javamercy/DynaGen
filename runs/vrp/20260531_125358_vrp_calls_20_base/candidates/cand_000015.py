import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    # Cheapest insertion construction
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    for cust in customers:
        best_inc = float('inf')
        best_route = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                inc = (distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] -
                       distance_matrix[route[pos-1], route[pos]])
                if inc < best_inc or (inc == best_inc and r_idx < best_route):
                    best_inc = inc
                    best_route = r_idx
                    best_pos = pos
        route = routes[best_route]
        routes[best_route] = route[:best_pos] + [cust] + route[best_pos:]
    report_best_vrp(routes)
    
    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in routes)
    
    # Tabu search
    max_iter = n * truck_count * 10
    tabu_list = {}  # (cust, src_route) -> remaining tenure
    tenure = 5
    for it in range(max_iter):
        best_move = None
        best_new_max = float('inf')
        best_new_total = float('inf')
        for cust in range(1, n):
            # Find current route and position of cust
            cur_route_idx = None
            cur_pos = None
            for r_idx, route in enumerate(routes):
                try:
                    cur_pos_tmp = route.index(cust)
                    cur_route_idx = r_idx
                    cur_pos = cur_pos_tmp
                    break
                except ValueError:
                    continue
            if cur_route_idx is None:
                continue
            # Remove cust from current route
            new_src = routes[cur_route_idx][:cur_pos] + routes[cur_route_idx][cur_pos+1:]
            src_dist = route_distance(new_src)
            # Consider all insertion positions in all routes
            for dst_route_idx, dst_route in enumerate(routes):
                for ins_pos in range(1, len(dst_route)):
                    if dst_route_idx == cur_route_idx and ins_pos == cur_pos:
                        continue
                    # Construct new destination route
                    new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                    dst_dist = route_distance(new_dst)
                    # Compute new max
                    new_max = src_dist if src_dist > dst_dist else dst_dist
                    for r_idx_other, r_other in enumerate(routes):
                        if r_idx_other == cur_route_idx:
                            continue
                        if r_idx_other == dst_route_idx:
                            continue
                        other_dist = route_distance(r_other)
                        if other_dist > new_max:
                            new_max = other_dist
                    total = src_dist + dst_dist + sum(
                        route_distance(r) for r in routes if r not in (new_src, new_dst))
                    # Check tabu
                    is_tabu = (cust, cur_route_idx) in tabu_list and tabu_list[(cust, cur_route_idx)] > 0
                    if is_tabu and new_max >= best_max:
                        continue
                    # Update best move if better than current best
                    if new_max < best_new_max or (new_max == best_new_max and total < best_new_total):
                        best_new_max = new_max
                        best_new_total = total
                        best_move = (cust, cur_route_idx, dst_route_idx, ins_pos, new_src, new_dst)
                    elif new_max == best_new_max and total == best_new_total:
                        # Tie-break: smallest customer index, then smallest source route
                        if cust < best_move[0] or (cust == best_move[0] and cur_route_idx < best_move[1]):
                            best_move = (cust, cur_route_idx, dst_route_idx, ins_pos, new_src, new_dst)
        if best_move is None:
            break
        # Apply best move
        cust, src_route_idx, dst_route_idx, ins_pos, new_src, new_dst = best_move
        if src_route_idx == dst_route_idx:
            routes[src_route_idx] = new_dst
        else:
            routes[src_route_idx] = new_src
            routes[dst_route_idx] = new_dst
        # Update tabu list
        tabu_list[(cust, src_route_idx)] = tenure + 1
        keys_to_delete = []
        for key in list(tabu_list.keys()):
            tabu_list[key] -= 1
            if tabu_list[key] <= 0:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del tabu_list[key]
        # Update best solution
        current_max = max(route_distance(r) for r in routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
    return best_routes
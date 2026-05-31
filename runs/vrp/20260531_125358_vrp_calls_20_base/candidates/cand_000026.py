import numpy as np
from collections import deque

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
        routes[best_route] = routes[best_route][:best_pos] + [cust] + routes[best_route][best_pos:]
    report_best_vrp(routes)

    # Balancing procedure (from cand_000014)
    def balance_routes(routes, lengths):
        improved = True
        while improved:
            improved = False
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_overall_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_insertion_len = float('inf')
                best_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_insertion_len:
                        best_insertion_len = l
                        best_pos = p
                new_min_route = min_route[:best_pos] + [cust] + min_route[best_pos:]
                new_min_len = route_distance(new_min_route)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_overall_reduction:
                    best_overall_reduction = reduction
                    best_cust = (cust, best_pos)
            if best_cust is not None:
                cust, best_insert_pos = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:best_insert_pos] + [cust] + min_route[best_insert_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_distance(new_max)
                lengths[min_idx] = route_distance(new_min)
                improved = True
                report_best_vrp(routes)
            else:
                break
        return routes, lengths

    lengths = [route_distance(r) for r in routes]
    routes, lengths = balance_routes([r[:] for r in routes], lengths[:])

    best_routes = [r[:] for r in routes]
    best_max = max(lengths)

    # Tabu search with relocate moves
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
                if dst_route_idx == cur_route_idx and len(dst_route) == 2:
                    continue  # avoid empty destination? actually source after removal might become empty
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
                        route_distance(r) for r_idx, r in enumerate(routes) if r_idx not in (cur_route_idx, dst_route_idx))
                    # Check tabu
                    is_tabu = (cust, cur_route_idx) in tabu_list and tabu_list[(cust, cur_route_idx)] > 0
                    if is_tabu and new_max >= best_max:
                        continue
                    # Update best move
                    if new_max < best_new_max or (new_max == best_new_max and total < best_new_total):
                        best_new_max = new_max
                        best_new_total = total
                        best_move = (cust, cur_route_idx, dst_route_idx, ins_pos)
                    elif new_max == best_new_max and total == best_new_total:
                        # Tie-break: smallest customer index, then smallest source route
                        if cust < best_move[0] or (cust == best_move[0] and cur_route_idx < best_move[1]):
                            best_move = (cust, cur_route_idx, dst_route_idx, ins_pos)
        if best_move is None:
            break
        # Apply best move
        cust, src_route_idx, dst_route_idx, ins_pos = best_move
        # Remove cust from src
        src_route = routes[src_route_idx]
        pos_src = src_route.index(cust)
        new_src = src_route[:pos_src] + src_route[pos_src+1:]
        routes[src_route_idx] = new_src
        # Insert into dst
        dst_route = routes[dst_route_idx]
        new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
        routes[dst_route_idx] = new_dst
        # Update tabu list
        tabu_list[(cust, src_route_idx)] = tenure
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
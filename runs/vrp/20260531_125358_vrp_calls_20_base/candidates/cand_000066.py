import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def two_opt_delta(route, i, j):
        # returns new route after reversing segment i..j and its length delta
        if j - i < 1:
            return route, 0.0
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        # compute delta: old edges (i-1,i) and (j,j+1) replaced by (i-1,j) and (i,j+1)
        old1 = distance_matrix[route[i-1], route[i]]
        old2 = distance_matrix[route[j], route[j+1]]
        new1 = distance_matrix[route[i-1], route[j]]
        new2 = distance_matrix[route[i], route[j+1]]
        delta = new1 + new2 - old1 - old2
        return new_route, delta
    
    def best_improving_move(routes, lengths):
        best_move = None
        best_new_max = max(lengths)
        best_new_total = sum(lengths)
        best_tie = (0, 0, 0)  # (max_reduction, total_reduction, -route_idx)
        current_max = best_new_max
        current_total = best_new_total
        n_cust = n - 1
        # Relocate moves
        for cust in range(1, n):
            src_idx = None
            src_pos = None
            for r_idx, route in enumerate(routes):
                if cust in route:
                    src_idx = r_idx
                    src_pos = route.index(cust)
                    break
            if src_idx is None:
                continue
            # delta for removing cust from src route
            src_route = routes[src_idx]
            if len(src_route) <= 2:
                continue
            removal_delta = distance_matrix[src_route[src_pos-1], src_route[src_pos]] + distance_matrix[src_route[src_pos], src_route[src_pos+1]] - distance_matrix[src_route[src_pos-1], src_route[src_pos+1]]
            for dst_idx in range(truck_count):
                if dst_idx == src_idx:
                    continue
                dst_route = routes[dst_idx]
                if len(dst_route) <= 1:
                    continue
                for ins_pos in range(1, len(dst_route)):
                    # delta for inserting cust at ins_pos
                    insertion_delta = distance_matrix[dst_route[ins_pos-1], cust] + distance_matrix[cust, dst_route[ins_pos]] - distance_matrix[dst_route[ins_pos-1], dst_route[ins_pos]]
                    new_src_len = lengths[src_idx] + removal_delta
                    new_dst_len = lengths[dst_idx] + insertion_delta
                    new_max = max(new_src_len, new_dst_len, max(lengths[:src_idx] + lengths[src_idx+1:dst_idx] + lengths[dst_idx+1:]) if truck_count > 2 else max(new_src_len, new_dst_len))
                    new_total = current_total + removal_delta + insertion_delta
                    max_reduction = current_max - new_max
                    total_reduction = current_total - new_total
                    tie = (max_reduction, total_reduction, -src_idx)
                    if tie > best_tie:
                        best_tie = tie
                        best_move = ('relocate', cust, src_idx, src_pos, dst_idx, ins_pos, new_src_len, new_dst_len)
                        best_new_max = new_max
                        best_new_total = new_total
                    elif tie == best_tie:
                        # break tie by route index
                        if src_idx < best_move[2] if best_move else True:
                            best_move = ('relocate', cust, src_idx, src_pos, dst_idx, ins_pos, new_src_len, new_dst_len)
        # Swap moves
        for i_idx in range(truck_count):
            i_route = routes[i_idx]
            if len(i_route) <= 2:
                continue
            for i_pos in range(1, len(i_route)-1):
                cust_i = i_route[i_pos]
                # delta for removing cust_i
                remove_i_delta = distance_matrix[i_route[i_pos-1], i_route[i_pos]] + distance_matrix[i_route[i_pos], i_route[i_pos+1]] - distance_matrix[i_route[i_pos-1], i_route[i_pos+1]]
                for j_idx in range(i_idx+1, truck_count):
                    j_route = routes[j_idx]
                    if len(j_route) <= 2:
                        continue
                    for j_pos in range(1, len(j_route)-1):
                        cust_j = j_route[j_pos]
                        # delta for removing cust_j
                        remove_j_delta = distance_matrix[j_route[j_pos-1], j_route[j_pos]] + distance_matrix[j_route[j_pos], j_route[j_pos+1]] - distance_matrix[j_route[j_pos-1], j_route[j_pos+1]]
                        # delta for inserting cust_i in j route at j_pos (after removal of j)
                        # careful: after removing j, the route segment changes. We need to compute insertion at j_pos (where j was)
                        # But we will construct new routes directly and compute distance? To avoid complexity, we'll compute delta approximately?
                        # Better to build new routes and compute lengths; but we can approximate insertion delta as if j wasn't there.
                        # For accuracy, we'll compute new lengths directly.
                        new_i_route = i_route[:i_pos] + i_route[i_pos+1:]
                        new_i_route = new_i_route[:i_pos] + [cust_j] + new_i_route[i_pos:]
                        new_j_route = j_route[:j_pos] + j_route[j_pos+1:]
                        new_j_route = new_j_route[:j_pos] + [cust_i] + new_j_route[j_pos:]
                        new_i_len = route_distance(new_i_route)
                        new_j_len = route_distance(new_j_route)
                        new_max = max(new_i_len, new_j_len, max(lengths[:i_idx] + lengths[i_idx+1:j_idx] + lengths[j_idx+1:]) if truck_count > 2 else max(new_i_len, new_j_len))
                        new_total = current_total - lengths[i_idx] - lengths[j_idx] + new_i_len + new_j_len
                        max_reduction = current_max - new_max
                        total_reduction = current_total - new_total
                        tie = (max_reduction, total_reduction, -i_idx)
                        if tie > best_tie:
                            best_tie = tie
                            best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i_route, new_j_route, new_i_len, new_j_len)
                            best_new_max = new_max
                            best_new_total = new_total
                        elif tie == best_tie:
                            if i_idx < best_move[2] if best_move else True:
                                best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i_route, new_j_route, new_i_len, new_j_len)
        # 2-opt moves
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route, delta = two_opt_delta(route, i, j)
                    if delta >= 0:
                        continue
                    new_len = lengths[r_idx] + delta
                    new_max = max(new_len, max(lengths[:r_idx] + lengths[r_idx+1:]) if truck_count > 1 else new_len)
                    new_total = current_total + delta
                    max_reduction = current_max - new_max
                    total_reduction = current_total - new_total
                    tie = (max_reduction, total_reduction, -r_idx)
                    if tie > best_tie:
                        best_tie = tie
                        best_move = ('2opt', r_idx, i, j, new_route, new_len)
                        best_new_max = new_max
                        best_new_total = new_total
        if best_move is not None and best_new_max < current_max:
            return best_move, best_new_max, best_new_total
        return None, current_max, current_total
    
    def perturbation(routes, lengths):
        # random swap with two_opt refinement and balancing
        perturb_iter = min(50, n * truck_count // 2)
        for _ in range(perturb_iter):
            usable = [i for i, r in enumerate(routes) if len(r) > 2]
            if len(usable) < 2:
                break
            r1 = random.choice(usable)
            r2 = random.choice([i for i in usable if i != r1])
            pos1 = random.randint(1, len(routes[r1])-2)
            pos2 = random.randint(1, len(routes[r2])-2)
            cust1 = routes[r1][pos1]
            cust2 = routes[r2][pos2]
            # Swap
            new_r1 = routes[r1][:pos1] + routes[r1][pos1+1:]
            new_r1 = new_r1[:pos1] + [cust2] + new_r1[pos1:]
            new_r2 = routes[r2][:pos2] + routes[r2][pos2+1:]
            new_r2 = new_r2[:pos2] + [cust1] + new_r2[pos2:]
            routes[r1] = new_r1
            routes[r2] = new_r2
            # Apply 2-opt to each if length > 2
            for idx in (r1, r2):
                if len(routes[idx]) > 2:
                    improved = True
                    max_opt_iter = 10
                    it = 0
                    while improved and it < max_opt_iter:
                        improved = False
                        it += 1
                        route = routes[idx]
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route, delta = two_opt_delta(route, i, j)
                                if delta < -1e-10:
                                    routes[idx] = new_route
                                    improved = True
                                    break
                            if improved:
                                break
            # Recompute lengths
            for idx in (r1, r2):
                lengths[idx] = route_distance(routes[idx])
            # Balance attempt
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx != min_idx:
                # try to move one customer from max to min if reduces max
                best_cust = None
                best_insert_pos = None
                best_reduction = 0
                for pos in range(1, len(routes[max_idx])-1):
                    cust = routes[max_idx][pos]
                    new_max_route = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                    new_max_len = route_distance(new_max_route)
                    for ins_pos in range(1, len(routes[min_idx])):
                        new_min_route = routes[min_idx][:ins_pos] + [cust] + routes[min_idx][ins_pos:]
                        new_min_len = route_distance(new_min_route)
                        other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                        new_global_max = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                        reduction = max(lengths) - new_global_max
                        if reduction > best_reduction:
                            best_reduction = reduction
                            best_cust = cust
                            best_insert_pos = ins_pos
                if best_cust is not None:
                    # apply move
                    new_max_route = [node for node in routes[max_idx] if node != best_cust]
                    new_min_route = routes[min_idx][:best_insert_pos] + [best_cust] + routes[min_idx][best_insert_pos:]
                    routes[max_idx] = new_max_route
                    routes[min_idx] = new_min_route
                    lengths[max_idx] = route_distance(new_max_route)
                    lengths[min_idx] = route_distance(new_min_route)
        return routes, lengths
    
    def regret_insertion_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                regret = incs[1][0] - incs[0][0] if len(incs) >= 2 else 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes
    
    best_routes = None
    best_max = float('inf')
    num_restarts = max(1, min(5, n // 10))
    for restart in range(num_restarts):
        routes = regret_insertion_construction()
        lengths = [route_distance(r) for r in routes]
        # initial balancing
        max_idx = max(range(truck_count), key=lambda i: lengths[i])
        min_idx = min(range(truck_count), key=lambda i: lengths[i])
        while max_idx != min_idx:
            best_cust = None
            best_insert_pos = None
            best_reduction = 0
            for pos in range(1, len(routes[max_idx])-1):
                cust = routes[max_idx][pos]
                new_max_route = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                new_max_len = route_distance(new_max_route)
                for ins_pos in range(1, len(routes[min_idx])):
                    new_min_route = routes[min_idx][:ins_pos] + [cust] + routes[min_idx][ins_pos:]
                    new_min_len = route_distance(new_min_route)
                    other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                    new_global_max = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                    reduction = max(lengths) - new_global_max
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_cust = cust
                        best_insert_pos = ins_pos
            if best_cust is not None:
                new_max_route = [node for node in routes[max_idx] if node != best_cust]
                new_min_route = routes[min_idx][:best_insert_pos] + [best_cust] + routes[min_idx][best_insert_pos:]
                routes[max_idx] = new_max_route
                routes[min_idx] = new_min_route
                lengths[max_idx] = route_distance(new_max_route)
                lengths[min_idx] = route_distance(new_min_route)
                max_idx = max(range(truck_count), key=lambda i: lengths[i])
                min_idx = min(range(truck_count), key=lambda i: lengths[i])
            else:
                break
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        
        max_iter = n * truck_count * 2
        for iteration in range(max_iter):
            move, new_max, new_total = best_improving_move(routes, lengths)
            if move is not None and new_max < current_max:
                if move[0] == 'relocate':
                    _, cust, src_idx, src_pos, dst_idx, ins_pos, new_src_len, new_dst_len = move
                    # apply
                    src_route = routes[src_idx]
                    src_route.pop(src_pos)
                    routes[dst_idx].insert(ins_pos, cust)
                    lengths[src_idx] = new_src_len
                    lengths[dst_idx] = new_dst_len
                elif move[0] == 'swap':
                    _, i_idx, i_pos, j_idx, j_pos, new_i_route, new_j_route, new_i_len, new_j_len = move
                    routes[i_idx] = new_i_route
                    routes[j_idx] = new_j_route
                    lengths[i_idx] = new_i_len
                    lengths[j_idx] = new_j_len
                elif move[0] == '2opt':
                    _, r_idx, i, j, new_route, new_len = move
                    routes[r_idx] = new_route
                    lengths[r_idx] = new_len
                current_max = max(lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
            else:
                # perturbation
                routes, lengths = perturbation(routes, lengths)
                current_max = max(lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
    return best_routes
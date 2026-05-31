import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def max_route_len(routes):
        return max(route_length(r) for r in routes) if routes else float('inf')
    
    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 10)
    
    for attempt in range(max_attempts):
        # Construction: regret heuristic
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        random.shuffle(unassigned)
        while unassigned:
            best_cust = None
            best_regret = -1e9
            best_data = None
            for cust in unassigned:
                insert_info = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_info.append((new_max, cost, r_idx, pos))
                if not insert_info:
                    continue
                insert_info.sort(key=lambda x: (x[0], x[1]))
                best = insert_info[0]
                second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                if best_cust is None or regret > best_regret or (regret == best_regret and best[1] < best_data[1]):
                    best_cust = cust
                    best_regret = regret
                    best_data = (best[0], best[1], best[2], best[3])
            if best_cust is None:
                break
            _, _, r_idx, pos = best_data
            routes[r_idx].insert(pos, best_cust)
            unassigned.remove(best_cust)
        
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        
        # Improvement: greedy only
        max_iter = n * truck_count
        iter_count = 0
        stagnation = 0
        perturb_size = 0.10
        max_perturb_size = 0.25
        perturb_inc = 0.05
        
        while iter_count < max_iter:
            improved = False
            # Inter-relocate
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            if len(max_route) > 2:
                best_move = None
                best_new_max = current_max
                for cust in max_route[1:-1]:
                    new_max_route = [x for x in max_route if x != cust]
                    new_max_len = route_length(new_max_route)
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other_route = routes[r_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                            candidate = max(new_max_len, new_other_len, *other_lens)
                            if candidate < best_new_max:
                                best_new_max = candidate
                                best_move = (cust, max_idx, r_idx, pos)
                if best_move:
                    cust, from_idx, to_idx, pos = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    current_max = best_new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                    improved = True
            # Inter-swap
            if not improved:
                lengths = [route_length(r) for r in routes]
                max_idx = int(np.argmax(lengths))
                max_route = routes[max_idx]
                if len(max_route) > 2:
                    best_move = None
                    best_new_max = current_max
                    for cust_i in max_route[1:-1]:
                        for other_idx in range(truck_count):
                            if other_idx == max_idx:
                                continue
                            other_route = routes[other_idx]
                            for cust_j in other_route[1:-1]:
                                new_max_route = [cust_j if x == cust_i else x for x in max_route]
                                new_other_route = [cust_i if x == cust_j else x for x in other_route]
                                new_max_len = route_length(new_max_route)
                                new_other_len = route_length(new_other_route)
                                other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, other_idx)]
                                candidate = max(new_max_len, new_other_len, *other_lens)
                                if candidate < best_new_max:
                                    best_new_max = candidate
                                    best_move = (cust_i, max_idx, cust_j, other_idx)
                    if best_move:
                        cust_i, idx1, cust_j, idx2 = best_move
                        routes[idx1] = [cust_j if x == cust_i else x for x in routes[idx1]]
                        routes[idx2] = [cust_i if x == cust_j else x for x in routes[idx2]]
                        current_max = best_new_max
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
                        improved = True
            # Intra-2opt
            if not improved:
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    best_delta = 0.0
                    best_pos = None
                    for i in range(1, len(route)-2):
                        for k in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            new_len = route_length(new_route)
                            delta = route_length(route) - new_len
                            if delta > best_delta + 1e-12:
                                best_delta = delta
                                best_pos = (i, k, r_idx)
                    if best_pos:
                        i, k, r_idx = best_pos
                        routes[r_idx] = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        current_max = max_route_len(routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
                        improved = True
            if improved:
                stagnation = 0
                perturb_size = 0.10
            else:
                stagnation += 1
                if stagnation >= 10:
                    # Ruin-recreate
                    route_lens = [(route_length(r), idx) for idx, r in enumerate(routes)]
                    route_lens.sort(reverse=True)
                    num_to_remove = max(1, int((n-1) * perturb_size))
                    removed = []
                    for _, r_idx in route_lens:
                        route = routes[r_idx]
                        if len(route) <= 2:
                            continue
                        can_remove = min(num_to_remove - len(removed), len(route)-2)
                        if can_remove <= 0:
                            break
                        remove_set = set(random.sample(route[1:-1], can_remove))
                        for cust in remove_set:
                            removed.append((r_idx, cust))
                        routes[r_idx] = [x for x in route if x not in remove_set]
                        if len(removed) >= num_to_remove:
                            break
                    unassigned = [cust for _, cust in removed]
                    random.shuffle(unassigned)
                    while unassigned:
                        best_cust = None
                        best_regret = -1e9
                        best_data = None
                        for cust in unassigned:
                            insert_info = []
                            for r_idx, route in enumerate(routes):
                                for pos in range(1, len(route)):
                                    prev = route[pos-1]
                                    nxt = route[pos]
                                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                    new_len = route_length(route) + cost
                                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                                    new_max = max(new_len, *other_lens)
                                    insert_info.append((new_max, cost, r_idx, pos))
                            if not insert_info:
                                continue
                            insert_info.sort(key=lambda x: (x[0], x[1]))
                            best = insert_info[0]
                            second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                            regret = second[0] - best[0]
                            if best_cust is None or regret > best_regret or (regret == best_regret and best[1] < best_data[1]):
                                best_cust = cust
                                best_regret = regret
                                best_data = (best[0], best[1], best[2], best[3])
                        if best_cust is None:
                            break
                        _, _, r_idx, pos = best_data
                        routes[r_idx].insert(pos, best_cust)
                        unassigned.remove(best_cust)
                    current_max = max_route_len(routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                    perturb_size = min(perturb_size + perturb_inc, max_perturb_size)
                    stagnation = 0
            iter_count += 1
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
    if best_routes is None:
        best_routes = routes
    return best_routes
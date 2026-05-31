import numpy as np
import random
from math import exp

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
    
    def regret2_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            candidates = []
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
                if insert_info:
                    insert_info.sort(key=lambda x: (x[0], x[1]))
                    best = insert_info[0]
                    second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                    regret = second[0] - best[0]
                    candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            # deterministic tie-breaking: lower cust index if equal regret and cost
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            chosen = candidates[0]
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        return routes
    
    def intra_2opt_improve(routes):
        improved = True
        while improved:
            improved = False
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                best_delta = 0.0
                best_ij = None
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_len = route_length(new_route)
                        if new_len < route_length(route):
                            delta = route_length(route) - new_len
                            if delta > best_delta:
                                best_delta = delta
                                best_ij = (i, k, r_idx)
                if best_ij:
                    i, k, r_idx = best_ij
                    routes[r_idx] = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    improved = True
        return routes
    
    def inter_relocate_best(routes, current_max, lengths):
        best_move = None
        best_new_max = current_max
        max_idx = int(np.argmax(lengths))
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            return False, None
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
                    new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                    if new_max_candidate < best_new_max - 1e-12:
                        best_new_max = new_max_candidate
                        best_move = (cust, max_idx, r_idx, pos)
        if best_move:
            cust, from_idx, to_idx, pos = best_move
            routes[from_idx] = [x for x in routes[from_idx] if x != cust]
            routes[to_idx].insert(pos, cust)
            return True, best_new_max
        return False, None
    
    def inter_swap_best(routes, current_max, lengths):
        best_move = None
        best_new_max = current_max
        max_idx = int(np.argmax(lengths))
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            return False, None
        for cust_i in max_route[1:-1]:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for cust_j in other_route[1:-1]:
                    new_max_route = [x if x != cust_i else cust_j for x in max_route]
                    new_other_route = [x if x != cust_j else cust_i for x in other_route]
                    new_max_len = route_length(new_max_route)
                    new_other_len = route_length(new_other_route)
                    other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, other_idx)]
                    new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                    if new_max_candidate < best_new_max - 1e-12:
                        best_new_max = new_max_candidate
                        best_move = (cust_i, max_idx, cust_j, other_idx)
        if best_move:
            cust_i, from_idx, cust_j, to_idx = best_move
            routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
            routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
            return True, best_new_max
        return False, None
    
    def intra_2opt_best(routes, current_max, lengths):
        best_move = None
        best_new_max = current_max
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for k in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    new_len = route_length(new_route)
                    new_max = max(new_len, *[route_length(routes[i]) for i in range(truck_count) if i != r_idx])
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_move = (i, k, r_idx)
        if best_move:
            i, k, r_idx = best_move
            routes[r_idx] = routes[r_idx][:i] + routes[r_idx][i:k+1][::-1] + routes[r_idx][k+1:]
            return True, best_new_max
        return False, None
    
    def ruin_recreate(routes, current_max):
        lengths = [route_length(r) for r in routes]
        max_idx = max(range(truck_count), key=lambda i: lengths[i])
        route = routes[max_idx]
        if len(route) <= 3:
            return
        # remove up to 20% of customers from longest route, at least 1, at most 3
        num_remove = min(max(1, int(0.2 * (len(route)-2))), 3)
        remove_set = set(random.sample(route[1:-1], num_remove))
        removed = []
        for cust in remove_set:
            removed.append(cust)
        routes[max_idx] = [x for x in route if x not in remove_set]
        unassigned = removed
        random.shuffle(unassigned)
        # reinsert with regret-2
        while unassigned:
            candidates = []
            for cust in unassigned:
                insert_info = []
                for r_idx, r in enumerate(routes):
                    for pos in range(1, len(r)):
                        prev = r[pos-1]
                        nxt = r[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(r) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_info.append((new_max, cost, r_idx, pos))
                if insert_info:
                    insert_info.sort(key=lambda x: (x[0], x[1]))
                    best = insert_info[0]
                    second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                    regret = second[0] - best[0]
                    candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            chosen = candidates[0]
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
    
    best_routes = None
    best_max = float('inf')
    # Multiple restarts: use 3 attempts
    for attempt in range(3):
        routes = regret2_construction()
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        # Intra-2opt until local optimum
        intra_2opt_improve(routes)
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        # VND main loop
        max_iter = n * truck_count * 5
        iter_count = 0
        stagnation = 0
        neighborhoods = [('inter_relocate', inter_relocate_best),
                         ('inter_swap', inter_swap_best),
                         ('intra_2opt', intra_2opt_best)]
        while iter_count < max_iter:
            improved_this_cycle = False
            for nh_name, nh_func in neighborhoods:
                lengths = [route_length(r) for r in routes]
                success, new_max = nh_func(routes, current_max, lengths)
                if success:
                    current_max = new_max
                    improved_this_cycle = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
            if improved_this_cycle:
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= 5:
                    ruin_recreate(routes, current_max)
                    current_max = max_route_len(routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                    stagnation = 0
            iter_count += 1
    if best_routes is None:
        best_routes = routes
    return best_routes
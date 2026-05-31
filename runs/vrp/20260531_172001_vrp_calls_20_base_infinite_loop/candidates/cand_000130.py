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
        # Construction: min-max greedy with regret tie-breaking
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
                insert_info.sort(key=lambda x: (x[0], x[1]))
                best = insert_info[0]
                second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            chosen = candidates[0]
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        
        # Improvement phase: deterministic VND with perturbation
        max_iter = n * truck_count * 5
        iter_count = 0
        stagnation = 0
        perturb_size = 0.10
        max_perturb_size = 0.30
        perturb_inc = 0.05
        
        while iter_count < max_iter:
            # VND: apply best improving move among neighborhoods, repeat until no improvement
            improved = True
            while improved:
                improved = False
                neighborhoods = ['intra_2opt', 'inter_relocate', 'inter_swap', 'intra_relocate', 'intra_swap']
                best_move = None
                best_new_max = current_max
                
                for nh in neighborhoods:
                    if nh == 'intra_2opt':
                        for r_idx in range(truck_count):
                            route = routes[r_idx]
                            if len(route) <= 3:
                                continue
                            for i in range(1, len(route)-2):
                                for k in range(i+1, len(route)-1):
                                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                                    new_len = route_length(new_route)
                                    if new_len < route_length(route):
                                        other_lens = [route_length(routes[j]) for j in range(truck_count) if j != r_idx]
                                        new_max_candidate = max(new_len, *other_lens)
                                        if new_max_candidate < best_new_max:
                                            best_new_max = new_max_candidate
                                            best_move = ('intra_2opt', r_idx, i, k)
                    elif nh == 'inter_relocate':
                        lengths = [route_length(r) for r in routes]
                        max_idx = int(np.argmax(lengths))
                        max_route = routes[max_idx]
                        if len(max_route) > 2:
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
                                        if new_max_candidate < best_new_max:
                                            best_new_max = new_max_candidate
                                            best_move = ('inter_relocate', cust, max_idx, r_idx, pos)
                    elif nh == 'inter_swap':
                        lengths = [route_length(r) for r in routes]
                        max_idx = int(np.argmax(lengths))
                        max_route = routes[max_idx]
                        if len(max_route) > 2:
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
                                        if new_max_candidate < best_new_max:
                                            best_new_max = new_max_candidate
                                            best_move = ('inter_swap', cust_i, max_idx, cust_j, other_idx)
                    elif nh == 'intra_relocate':
                        for r_idx in range(truck_count):
                            route = routes[r_idx]
                            if len(route) <= 3:
                                continue
                            for i in range(1, len(route)-1):
                                for j in range(1, len(route)-1):
                                    if i == j:
                                        continue
                                    cust = route[i]
                                    new_route = route[:i] + route[i+1:]
                                    new_route = new_route[:j] + [cust] + new_route[j:]
                                    new_len = route_length(new_route)
                                    if new_len < route_length(route):
                                        other_lens = [route_length(routes[k]) for k in range(truck_count) if k != r_idx]
                                        new_max_candidate = max(new_len, *other_lens)
                                        if new_max_candidate < best_new_max:
                                            best_new_max = new_max_candidate
                                            best_move = ('intra_relocate', r_idx, i, j)
                    elif nh == 'intra_swap':
                        for r_idx in range(truck_count):
                            route = routes[r_idx]
                            if len(route) <= 3:
                                continue
                            for i in range(1, len(route)-1):
                                for j in range(i+1, len(route)-1):
                                    new_route = route[:]
                                    new_route[i], new_route[j] = new_route[j], new_route[i]
                                    new_len = route_length(new_route)
                                    if new_len < route_length(route):
                                        other_lens = [route_length(routes[k]) for k in range(truck_count) if k != r_idx]
                                        new_max_candidate = max(new_len, *other_lens)
                                        if new_max_candidate < best_new_max:
                                            best_new_max = new_max_candidate
                                            best_move = ('intra_swap', r_idx, i, j)
                
                if best_move is not None and best_new_max < current_max:
                    # apply best move
                    if best_move[0] == 'intra_2opt':
                        _, r_idx, i, k = best_move
                        routes[r_idx] = routes[r_idx][:i] + routes[r_idx][i:k+1][::-1] + routes[r_idx][k+1:]
                    elif best_move[0] == 'inter_relocate':
                        _, cust, from_idx, to_idx, pos = best_move
                        routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                        routes[to_idx].insert(pos, cust)
                    elif best_move[0] == 'inter_swap':
                        _, cust_i, from_idx, cust_j, to_idx = best_move
                        routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                        routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                    elif best_move[0] == 'intra_relocate':
                        _, r_idx, i, j = best_move
                        route = routes[r_idx]
                        cust = route[i]
                        route = route[:i] + route[i+1:]
                        route = route[:j] + [cust] + route[j:]
                        routes[r_idx] = route
                    elif best_move[0] == 'intra_swap':
                        _, r_idx, i, j = best_move
                        route = routes[r_idx]
                        route[i], route[j] = route[j], route[i]
                        routes[r_idx] = route
                    current_max = best_new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                    improved = True
            
            iter_count += 1
            if improved_from_perturb := False:  # placeholder, actually check if perturbation improved
                pass
            # Perturbation if no improvement after VND
            # Ruin-recreate from best solution
            if best_routes is not None:
                routes = [r[:] for r in best_routes]
                current_max = best_max
            # Increase stagnation and perturb_size
            stagnation += 1
            if stagnation >= 5:
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
                    best_regret = -1.0
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
        
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
    
    if best_routes is None:
        best_routes = routes
    return best_routes
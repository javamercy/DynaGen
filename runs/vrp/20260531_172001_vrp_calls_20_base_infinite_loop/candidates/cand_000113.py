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
    
    best_routes = None
    best_max = float('inf')
    max_restarts = max(1, n // 20)
    
    for restart in range(max_restarts):
        # Stochastic construction with softmax regret
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        temperature = 1.0  # higher -> more random
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
                candidates.append((regret, best[1], best[2], best[3], cust))
            # Softmax selection based on regret
            reg_vals = [c[0] for c in candidates]
            min_reg = min(reg_vals)
            weights = [exp((r - min_reg)/temperature) for r in reg_vals]
            total_w = sum(weights)
            probs = [w/total_w for w in weights]
            idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
            _, cost, r_idx, pos, cust = candidates[idx]
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
            temperature *= 0.99  # anneal
        
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        
        # Improvement with simulated annealing acceptance
        neighborhoods = ['inter_relocate', 'inter_swap', 'intra_2opt']
        nh_success = {nh: 0.0 for nh in neighborhoods}
        stagnation = 0
        perturb_size = 0.10
        max_perturb_size = 0.30
        perturb_inc = 0.05
        max_iter = n * truck_count * 2
        iter_count = 0
        T = 1.0  # initial temperature for SA
        T_min = 0.01
        alpha = 0.99
        
        while iter_count < max_iter and T > T_min:
            # Select neighborhood via softmax on success counts
            if any(nh_success.values()):
                success_vals = [nh_success[nh] for nh in neighborhoods]
                probs = [exp(s) for s in success_vals]
                total = sum(probs)
                probs = [p/total for p in probs]
                nh_choice = random.choices(neighborhoods, weights=probs, k=1)[0]
            else:
                nh_choice = random.choice(neighborhoods)
            
            improved_this_iter = False
            
            if nh_choice == 'inter_relocate':
                lengths = [route_length(r) for r in routes]
                max_idx = int(np.argmax(lengths))
                max_route = routes[max_idx]
                if len(max_route) > 2:
                    best_delta = 0.0
                    best_move = None
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
                                if new_max_candidate < current_max - 1e-12:
                                    delta = current_max - new_max_candidate
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_move = (cust, max_idx, r_idx, pos, new_max_candidate)
                    if best_move and best_delta > 0:
                        cust, from_idx, to_idx, pos, new_max_val = best_move
                        routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                        routes[to_idx].insert(pos, cust)
                        current_max = new_max_val
                        improved_this_iter = True
                    elif best_move:  # accept with probability (no improvement)
                        if random.random() < exp((current_max - new_max_val)/T):
                            cust, from_idx, to_idx, pos, new_max_val = best_move
                            routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                            routes[to_idx].insert(pos, cust)
                            current_max = new_max_val
                            improved_this_iter = True
            elif nh_choice == 'inter_swap':
                lengths = [route_length(r) for r in routes]
                max_idx = int(np.argmax(lengths))
                max_route = routes[max_idx]
                if len(max_route) > 2:
                    best_delta = 0.0
                    best_move = None
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
                                if new_max_candidate < current_max:
                                    delta = current_max - new_max_candidate
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_move = (cust_i, max_idx, cust_j, other_idx, new_max_candidate)
                    if best_move and best_delta > 0:
                        cust_i, from_idx, cust_j, to_idx, new_max_val = best_move
                        routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                        routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                        current_max = new_max_val
                        improved_this_iter = True
                    elif best_move:
                        if random.random() < exp((current_max - new_max_val)/T):
                            cust_i, from_idx, cust_j, to_idx, new_max_val = best_move
                            routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                            routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                            current_max = new_max_val
                            improved_this_iter = True
            else:  # intra_2opt
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
                    if best_ij and best_delta > 0:
                        i, k, r_idx = best_ij
                        routes[r_idx] = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_max = max_route_len(routes)
                        if new_max < current_max:
                            current_max = new_max
                            improved_this_iter = True
                    elif best_ij:
                        i, k, r_idx = best_ij
                        old_len = route_length(route)
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_len = route_length(new_route)
                        if random.random() < exp((old_len - new_len)/T):
                            routes[r_idx] = new_route
                            new_max = max_route_len(routes)
                            if new_max < current_max:
                                current_max = new_max
                            improved_this_iter = True
            
            if improved_this_iter:
                nh_success[nh_choice] += 1
                stagnation = 0
                perturb_size = 0.10
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(routes)
            else:
                stagnation += 1
                T *= alpha
                if stagnation >= 15:
                    # Ruin-recreate perturbation with randomness
                    route_lens = [(route_length(r), idx) for idx, r in enumerate(routes)]
                    route_lens.sort(reverse=True)
                    num_to_remove = max(1, int((n-1) * perturb_size))
                    removed = []
                    # Randomly select routes to remove from (not just longest)
                    candidate_routes = list(range(truck_count))
                    random.shuffle(candidate_routes)
                    for r_idx in candidate_routes:
                        route = routes[r_idx]
                        if len(route) <= 2:
                            continue
                        can_remove = min(num_to_remove - len(removed), len(route)-2)
                        if can_remove <= 0:
                            break
                        # Remove random customers
                        remove_set = set(random.sample(route[1:-1], can_remove))
                        for cust in remove_set:
                            removed.append((r_idx, cust))
                        routes[r_idx] = [x for x in route if x not in remove_set]
                        if len(removed) >= num_to_remove:
                            break
                    # Reinsert using stochastic regret
                    unassigned = [cust for _, cust in removed]
                    random.shuffle(unassigned)
                    temp_ins = 1.0
                    while unassigned:
                        candidates_ins = []
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
                            candidates_ins.append((regret, best[1], best[2], best[3], cust))
                        if not candidates_ins:
                            break
                        reg_vals_ins = [c[0] for c in candidates_ins]
                        min_reg_ins = min(reg_vals_ins)
                        weights_ins = [exp((r - min_reg_ins)/temp_ins) for r in reg_vals_ins]
                        total_w_ins = sum(weights_ins)
                        probs_ins = [w/total_w_ins for w in weights_ins]
                        idx_ins = random.choices(range(len(candidates_ins)), weights=probs_ins, k=1)[0]
                        _, _, r_idx, pos, cust = candidates_ins[idx_ins]
                        routes[r_idx].insert(pos, cust)
                        unassigned.remove(cust)
                        temp_ins *= 0.99
                    current_max = max_route_len(routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                    perturb_size = min(perturb_size + perturb_inc, max_perturb_size)
                    stagnation = 0
                    nh_success = {nh: 0.0 for nh in neighborhoods}
                    T = max(T_min, T * alpha)
            
            iter_count += 1
        
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
    
    if best_routes is None:
        best_routes = routes
    return best_routes
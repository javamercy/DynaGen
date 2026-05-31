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
    max_attempts = max(1, n // 5)  # more attempts for diversity
    
    for attempt in range(max_attempts):
        # Randomly shuffle customers to create diverse initial solutions
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(customers)
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
            # Add some randomness: sometimes pick not the best
            if random.random() < 0.1:
                random.shuffle(candidates)
                chosen = candidates[0]
            else:
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
        
        neighborhoods = ['inter_relocate', 'inter_swap', 'intra_2opt']
        nh_success = {nh: 0.0 for nh in neighborhoods}
        stagnation = 0
        perturb_size = 0.20  # larger initial perturbation
        max_perturb_size = 0.50
        perturb_inc = 0.10
        max_iter = n * truck_count * 3  # more iterations
        iter_count = 0
        initial_temp = current_max * 0.3  # higher initial temp
        if initial_temp < 1e-12:
            initial_temp = 1.0
        cooling_rate_base = 0.995  # slower cooling
        cooling_rate = cooling_rate_base
        
        while iter_count < max_iter:
            T = initial_temp * (cooling_rate ** iter_count)
            if T < 1e-12:
                T = 1e-12
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
                    moves = []
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
                                delta = new_max_candidate - current_max
                                moves.append((delta, cust, max_idx, r_idx, pos, new_max_candidate))
                    if moves:
                        moves.sort(key=lambda x: x[0])
                        # Accept best improvement or probabilistic
                        best_move = moves[0]
                        if best_move[0] < 0:
                            _, cust, from_idx, to_idx, pos, new_max_val = best_move
                            routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                            routes[to_idx].insert(pos, cust)
                            current_max = new_max_val
                            improved_this_iter = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                        else:
                            for move in moves:
                                if move[0] >= 0:
                                    prob = exp(-move[0] / T)
                                    if random.random() < prob:
                                        _, cust, from_idx, to_idx, pos, new_max_val = move
                                        routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                                        routes[to_idx].insert(pos, cust)
                                        current_max = new_max_val
                                        improved_this_iter = True
                                        break
            elif nh_choice == 'inter_swap':
                lengths = [route_length(r) for r in routes]
                max_idx = int(np.argmax(lengths))
                max_route = routes[max_idx]
                if len(max_route) > 2:
                    moves = []
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
                                delta = new_max_candidate - current_max
                                moves.append((delta, cust_i, max_idx, cust_j, other_idx, new_max_candidate))
                    if moves:
                        moves.sort(key=lambda x: x[0])
                        best_move = moves[0]
                        if best_move[0] < 0:
                            _, cust_i, from_idx, cust_j, to_idx, new_max_val = best_move
                            routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                            routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                            current_max = new_max_val
                            improved_this_iter = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                        else:
                            for move in moves:
                                if move[0] >= 0:
                                    prob = exp(-move[0] / T)
                                    if random.random() < prob:
                                        _, cust_i, from_idx, cust_j, to_idx, new_max_val = move
                                        routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                                        routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                                        current_max = new_max_val
                                        improved_this_iter = True
                                        break
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
                            delta = new_len - route_length(route)
                            if delta < best_delta:
                                best_delta = delta
                                best_ij = (i, k, r_idx)
                    if best_ij:
                        i, k, r_idx = best_ij
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        routes[r_idx] = new_route
                        new_max = max_route_len(routes)
                        if new_max < current_max:
                            current_max = new_max
                            improved_this_iter = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                        else:
                            delta = new_max - current_max
                            if random.random() < exp(-delta / T):
                                current_max = new_max
                                improved_this_iter = True
            
            if improved_this_iter:
                nh_success[nh_choice] += 1
                stagnation = 0
                perturb_size = 0.20
                cooling_rate = cooling_rate_base
            else:
                stagnation += 1
                if stagnation >= 15:
                    # Stronger perturbation: ruin and recreate
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
                    nh_success = {nh: 0.0 for nh in neighborhoods}
            
            iter_count += 1
        
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
    
    if best_routes is None:
        best_routes = routes
    return best_routes
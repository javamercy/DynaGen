import numpy as np
import random
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    start_time = time.time()
    max_time = 170
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def max_route_len(routes):
        return max(route_length(r) for r in routes)
    
    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 5)  # increased attempts for intensive search
    
    for attempt in range(max_attempts):
        if time.time() - start_time > max_time:
            break
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
                # tie-breaking: prefer smaller max, larger regret, smaller cost, and customer index
                candidates.append((best[0], -regret, -best[1], cust, best[2], best[3]))
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[3]))
            chosen = candidates[0]
            new_max_val, _, _, cust, r_idx, pos = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        
        # Improvement phase
        lengths = [route_length(r) for r in routes]
        current_max = max(lengths)
        improved = True
        iter_count = 0
        max_iter = n * truck_count * 4  # increased max iterations
        stagnation = 0
        perturbation_threshold = 8
        
        neighborhoods = ['inter_relocate', 'inter_swap', 'intra_2opt', 'intra_oropt']
        
        while iter_count < max_iter:
            if time.time() - start_time > max_time:
                break
            improved_this_iter = False
            for nh in neighborhoods:
                if nh == 'inter_relocate':
                    lengths = [route_length(r) for r in routes]
                    max_idx = np.argmax(lengths)
                    max_route = routes[max_idx]
                    if len(max_route) > 2:
                        best_delta = 0
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
                                    new_max_candidate = max(new_max_len, new_other_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)])
                                    if new_max_candidate < current_max - 1e-12:
                                        delta = current_max - new_max_candidate
                                        if delta > best_delta:
                                            best_delta = delta
                                            best_move = (cust, max_idx, r_idx, pos)
                        if best_move is not None:
                            cust, from_idx, to_idx, pos = best_move
                            routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                            routes[to_idx].insert(pos, cust)
                            current_max = current_max - best_delta
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                            improved_this_iter = True
                elif nh == 'inter_swap':
                    lengths = [route_length(r) for r in routes]
                    max_idx = np.argmax(lengths)
                    max_route = routes[max_idx]
                    if len(max_route) > 2:
                        best_delta = 0
                        best_move = None
                        for cust_i in max_route[1:-1]:
                            for other_idx in range(truck_count):
                                if other_idx == max_idx:
                                    continue
                                other_route = routes[other_idx]
                                for cust_j in other_route[1:-1]:
                                    # Swap cust_i and cust_j
                                    new_max_route = [cust_j if x == cust_i else x for x in max_route]
                                    new_other_route = [cust_i if x == cust_j else x for x in other_route]
                                    new_max_len = route_length(new_max_route)
                                    new_other_len = route_length(new_other_route)
                                    new_max_candidate = max(new_max_len, new_other_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, other_idx)])
                                    if new_max_candidate < current_max - 1e-12:
                                        delta = current_max - new_max_candidate
                                        if delta > best_delta:
                                            best_delta = delta
                                            best_move = (cust_i, max_idx, cust_j, other_idx)
                        if best_move is not None:
                            cust_i, from_idx, cust_j, to_idx = best_move
                            routes[from_idx] = [cust_j if x == cust_i else x for x in routes[from_idx]]
                            routes[to_idx] = [cust_i if x == cust_j else x for x in routes[to_idx]]
                            current_max = current_max - best_delta
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                            improved_this_iter = True
                elif nh == 'intra_2opt':
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        best_delta = 0
                        best_ij = None
                        for i in range(1, len(route)-2):
                            for k in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                                new_len = route_length(new_route)
                                if new_len < route_length(route) - 1e-12:
                                    delta = route_length(route) - new_len
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_ij = (i, k, r_idx)
                        if best_ij is not None:
                            i, k, r_idx = best_ij
                            routes[r_idx] = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            new_max = max_route_len(routes)
                            if new_max < current_max:
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                            improved_this_iter = True
                elif nh == 'intra_oropt':
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        best_delta = 0
                        best_move = None
                        for i in range(1, len(route)-1):  # start of segment
                            for k in range(i, len(route)-1):  # end of segment (inclusive)
                                segment = route[i:k+1]
                                remaining = route[:i] + route[k+1:]
                                for pos in range(1, len(remaining)):
                                    new_route = remaining[:pos] + segment + remaining[pos:]
                                    new_len = route_length(new_route)
                                    if new_len < route_length(route) - 1e-12:
                                        delta = route_length(route) - new_len
                                        if delta > best_delta:
                                            best_delta = delta
                                            best_move = (r_idx, i, k, pos)
                        if best_move is not None:
                            r_idx, i, k, pos = best_move
                            route = routes[r_idx]
                            segment = route[i:k+1]
                            remaining = route[:i] + route[k+1:]
                            routes[r_idx] = remaining[:pos] + segment + remaining[pos:]
                            new_max = max_route_len(routes)
                            if new_max < current_max:
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                            improved_this_iter = True
                if improved_this_iter:
                    stagnation = 0
                    break  # restart neighborhood loop
            
            if not improved_this_iter:
                stagnation += 1
                if stagnation >= perturbation_threshold:
                    # Directed perturbation: move some customers from max route to underutilized routes
                    lengths = [route_length(r) for r in routes]
                    max_idx = np.argmax(lengths)
                    max_route = routes[max_idx]
                    # Identify underutilized routes: those with length less than average
                    avg_len = sum(lengths) / truck_count
                    candidates_routes = [i for i in range(truck_count) if i != max_idx and lengths[i] < avg_len]
                    if not candidates_routes:
                        candidates_routes = [i for i in range(truck_count) if i != max_idx]
                    num_moves = max(1, min(n // 10, len(max_route)-2))
                    for _ in range(num_moves):
                        if len(max_route) <= 2:
                            break
                        cust = random.choice(max_route[1:-1])
                        # Relocate this customer to the best underutilized route
                        best_route_idx = None
                        best_new_max = current_max
                        for r_idx in candidates_routes:
                            other_route = routes[r_idx]
                            for pos in range(1, len(other_route)+1):
                                new_other = other_route[:pos] + [cust] + other_route[pos:]
                                new_max_route = [x for x in max_route if x != cust]
                                new_max_len = route_length(new_max_route)
                                new_other_len = route_length(new_other)
                                other_lengths = lengths.copy()
                                other_lengths[max_idx] = new_max_len
                                other_lengths[r_idx] = new_other_len
                                candidate_max = max(other_lengths)
                                if candidate_max < best_new_max:
                                    best_new_max = candidate_max
                                    best_route_idx = r_idx
                                    best_pos = pos
                        if best_route_idx is not None:
                            routes[max_idx] = [x for x in routes[max_idx] if x != cust]
                            routes[best_route_idx].insert(best_pos, cust)
                            lengths[max_idx] = route_length(routes[max_idx])
                            lengths[best_route_idx] = route_length(routes[best_route_idx])
                            current_max = max(lengths)
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                    stagnation = 0
            
            iter_count += 1
            if not improved_this_iter and stagnation > 0:
                if iter_count >= max_iter:
                    break
        
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
    
    if best_routes is None:
        best_routes = routes
    return best_routes
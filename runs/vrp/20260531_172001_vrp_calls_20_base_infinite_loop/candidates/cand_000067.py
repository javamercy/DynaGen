import numpy as np

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
        return max(route_length(r) for r in routes)
    
    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 8)
    
    for attempt in range(max_attempts):
        # Deterministic regret-2 construction
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            candidates = []
            for cust in unassigned:
                best_cost = float('inf')
                best_second_cost = float('inf')
                best_r_idx = -1
                best_pos = -1
                second_r_idx = -1
                second_pos = -1
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        if new_max < best_cost or (new_max == best_cost and (cost, r_idx, pos) < (best_cost_orig, best_r_idx, best_pos)):
                            second_cost, second_r_idx, second_pos = best_cost, best_r_idx, best_pos
                            best_cost, best_r_idx, best_pos = new_max, r_idx, pos
                            best_cost_orig = cost
                        elif new_max < second_cost or (new_max == second_cost and (cost, r_idx, pos) < (second_cost_orig, second_r_idx, second_pos)):
                            second_cost, second_r_idx, second_pos = new_max, r_idx, pos
                            second_cost_orig = cost
                regret = best_cost - second_cost if second_cost != float('inf') else 0
                candidates.append((best_cost, -regret, -best_cost_orig, best_r_idx, best_pos, cust))
            candidates.sort(key=lambda x: (x[0], x[1]))
            chosen = candidates[0]
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        
        # Local search
        improved = True
        iter_count = 0
        max_iter = n * truck_count
        no_improve_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            # Inter-route relocate from longest route
            if len(max_route) > 2:
                best_delta = 0
                best_move = None
                for i, cust in enumerate(max_route[1:-1]):
                    new_max_route = [x for x in max_route if x != cust]
                    new_max_len = route_length(new_max_route)
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other_route = routes[r_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            others = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *others)
                            if new_max_candidate < current_max:
                                delta = current_max - new_max_candidate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (cust, max_idx, r_idx, pos, new_max_candidate)
                if best_move:
                    cust, from_idx, to_idx, pos, new_max_val = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    current_max = new_max_val
                    improved = True
                    no_improve_count = 0
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
            
            # Intra-route 2-opt
            if not improved:
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    improved_intra = False
                    for i in range(1, len(route)-2):
                        for k in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            new_len = route_length(new_route)
                            if new_len < route_length(route):
                                # Compute new max quickly
                                route[:] = new_route
                                improved_intra = True
                                break
                        if improved_intra:
                            break
                    if improved_intra:
                        improved = True
                        no_improve_count = 0
                        new_max = max_route_len(routes)
                        if new_max < current_max:
                            current_max = new_max
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= 10:
                    # Ruin-recreate: remove 20% customers from longest route only, reinsert deterministically
                    # Find longest route index
                    lengths = [route_length(r) for r in routes]
                    max_idx = int(np.argmax(lengths))
                    max_route = routes[max_idx]
                    if len(max_route) > 2:
                        num_remove = max(1, int(len(max_route[1:-1]) * 0.2))
                        # Remove the 'num_remove' customers with largest insertion cost? Sort customers by contribution to route length
                        # Simple: remove first num_remove customers (excluding ends) to avoid bias. Could randomize but deterministic: remove from positions 1..len-2
                        remove_indices = list(range(1, len(max_route)-1))
                        remove_indices.sort()  # deterministic
                        remove_indices = remove_indices[:num_remove]
                        removed_customers = [max_route[i] for i in remove_indices]
                        # Remove them
                        routes[max_idx] = [x for x in max_route if x not in removed_customers]
                        # Reinsert using deterministic regret-2
                        unassigned = removed_customers
                        while unassigned:
                            candidates = []
                            for cust in unassigned:
                                best_cost = float('inf')
                                best_second_cost = float('inf')
                                best_r_idx = -1
                                best_pos = -1
                                second_r_idx = -1
                                second_pos = -1
                                for r_idx, route in enumerate(routes):
                                    for pos in range(1, len(route)):
                                        prev = route[pos-1]
                                        nxt = route[pos]
                                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                        new_len = route_length(route) + cost
                                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                                        new_max = max(new_len, *other_lens)
                                        if new_max < best_cost or (new_max == best_cost and (cost, r_idx, pos) < (best_cost_orig, best_r_idx, best_pos)):
                                            second_cost, second_r_idx, second_pos = best_cost, best_r_idx, best_pos
                                            best_cost, best_r_idx, best_pos = new_max, r_idx, pos
                                            best_cost_orig = cost
                                        elif new_max < second_cost or (new_max == second_cost and (cost, r_idx, pos) < (second_cost_orig, second_r_idx, second_pos)):
                                            second_cost, second_r_idx, second_pos = new_max, r_idx, pos
                                            second_cost_orig = cost
                                regret = best_cost - second_cost if second_cost != float('inf') else 0
                                candidates.append((best_cost, -regret, -best_cost_orig, best_r_idx, best_pos, cust))
                            candidates.sort(key=lambda x: (x[0], x[1]))
                            chosen = candidates[0]
                            _, _, _, r_idx, pos, cust = chosen
                            routes[r_idx].insert(pos, cust)
                            unassigned.remove(cust)
                        current_max = max_route_len(routes)
                        no_improve_count = 0
                        improved = True
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
        # End of local search
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
    
    if best_routes is None:
        best_routes = routes
    return best_routes
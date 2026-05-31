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
    
    # Only one attempt but with randomization
    max_attempts = 1
    for _ in range(max_attempts):
        # Randomized greedy insertion: shuffle customers first
        unassigned = list(range(1, n))
        random.shuffle(unassigned)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in unassigned:
            # Find best insertion for this customer (minimum max route distance)
            best_info = None
            best_max_val = float('inf')
            best_cost = float('inf')
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max_val = max(new_len, *other_lens)
                    if new_max_val < best_max_val or (new_max_val == best_max_val and cost < best_cost):
                        best_max_val = new_max_val
                        best_cost = cost
                        best_info = (r_idx, pos)
            if best_info:
                r_idx, pos = best_info
                routes[r_idx].insert(pos, cust)
        
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        
        # Simulated annealing parameters
        avg_route_len = sum(route_length(r) for r in routes) / truck_count
        initial_temp = avg_route_len * 0.1
        if initial_temp < 1e-12:
            initial_temp = 1.0
        cooling_rate = 0.99
        max_iter = max(n * truck_count * 10, 2000)  # bounded by instance size
        iter_count = 0
        stagnation = 0
        
        neighborhoods = ['inter_relocate', 'inter_swap', 'intra_2opt']
        success_counts = {nh: 0.0 for nh in neighborhoods}
        
        while iter_count < max_iter:
            T = initial_temp * (cooling_rate ** iter_count)
            if T < 1e-12:
                T = 1e-12
            
            # Choose neighborhood based on success (softmax)
            if any(success_counts.values()):
                exp_vals = [exp(s) for s in success_counts.values()]
                total = sum(exp_vals)
                probs = [p/total for p in exp_vals]
                nh_choice = random.choices(neighborhoods, weights=probs, k=1)[0]
            else:
                nh_choice = random.choice(neighborhoods)
            
            improved = False
            
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
                            other = routes[r_idx]
                            for pos in range(1, len(other)):
                                new_other = other[:pos] + [cust] + other[pos:]
                                new_other_len = route_length(new_other)
                                other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                                candidate_max = max(new_max_len, new_other_len, *other_lens)
                                if candidate_max < current_max - 1e-12:
                                    delta = current_max - candidate_max
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_move = (cust, max_idx, r_idx, pos, candidate_max)
                    if best_move:
                        cust, from_idx, to_idx, pos, new_max = best_move
                        routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                        routes[to_idx].insert(pos, cust)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                        else:
                            delta = new_max - current_max
                            if random.random() < exp(-delta / T):
                                current_max = new_max
                                improved = True
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
                                candidate_max = max(new_max_len, new_other_len, *other_lens)
                                if candidate_max < current_max - 1e-12:
                                    delta = current_max - candidate_max
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_move = (cust_i, max_idx, cust_j, other_idx, candidate_max)
                    if best_move:
                        cust_i, from_idx, cust_j, to_idx, new_max = best_move
                        routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                        routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                        else:
                            delta = new_max - current_max
                            if random.random() < exp(-delta / T):
                                current_max = new_max
                                improved = True
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
                            if new_len < route_length(route) - 1e-12:
                                delta = route_length(route) - new_len
                                if delta > best_delta:
                                    best_delta = delta
                                    best_ij = (i, k, r_idx)
                    if best_ij:
                        i, k, r_idx = best_ij
                        routes[r_idx] = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_max = max_route_len(routes)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                        else:
                            delta = new_max - current_max
                            if random.random() < exp(-delta / T):
                                current_max = new_max
                                improved = True
            
            if improved:
                success_counts[nh_choice] += 1
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= 10:
                    # Restart: reset temperature based on current state
                    avg_route_len = sum(route_length(r) for r in routes) / truck_count
                    initial_temp = avg_route_len * 0.1
                    if initial_temp < 1e-12:
                        initial_temp = 1.0
                    cooling_rate = 0.99
                    # Perturb: remove and reinsert a few customers randomly
                    route_lens = [(route_length(r), idx) for idx, r in enumerate(routes)]
                    route_lens.sort(reverse=True)
                    num_to_remove = max(1, int((n-1) * 0.1))
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
                    for cust in unassigned:
                        best_info = None
                        best_max_val = float('inf')
                        best_cost = float('inf')
                        for r_idx in range(truck_count):
                            route = routes[r_idx]
                            for pos in range(1, len(route)):
                                prev = route[pos-1]
                                nxt = route[pos]
                                cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                new_len = route_length(route) + cost
                                other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                                new_max_val = max(new_len, *other_lens)
                                if new_max_val < best_max_val or (new_max_val == best_max_val and cost < best_cost):
                                    best_max_val = new_max_val
                                    best_cost = cost
                                    best_info = (r_idx, pos)
                        if best_info:
                            r_idx, pos = best_info
                            routes[r_idx].insert(pos, cust)
                    current_max = max_route_len(routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                    stagnation = 0
                    success_counts = {nh: 0.0 for nh in neighborhoods}
            
            iter_count += 1
    
    if best_routes is None:
        best_routes = routes
    return best_routes
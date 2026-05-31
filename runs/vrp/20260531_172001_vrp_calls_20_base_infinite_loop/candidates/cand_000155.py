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
    
    # Greedy cheapest insertion (min max)
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    while unassigned:
        best_cust = None
        best_cost = float('inf')
        best_r_idx = -1
        best_pos = -1
        for cust in unassigned:
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost_inc = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost_inc
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if new_max < best_cost:
                        best_cost = new_max
                        best_cust = cust
                        best_r_idx = r_idx
                        best_pos = pos
        if best_cust is None:
            break
        routes[best_r_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    best_routes = [r[:] for r in routes]
    best_max = max_route_len(routes)
    report_best_vrp(routes)
    
    # Simulated annealing
    current_routes = [r[:] for r in routes]
    current_max = best_max
    initial_temp = best_max * 0.2
    if initial_temp < 1e-12:
        initial_temp = 1.0
    cooling_rate = 0.995
    max_iter = n * truck_count * 2
    stagnation = 0
    neighborhoods = ['inter_relocate', 'intra_2opt']
    
    for iteration in range(max_iter):
        T = initial_temp * (cooling_rate ** iteration)
        if T < 1e-12:
            T = 1e-12
        
        nh = random.choice(neighborhoods)
        improved = False
        
        if nh == 'inter_relocate':
            lengths = [route_length(r) for r in current_routes]
            max_idx = int(np.argmax(lengths))
            max_route = current_routes[max_idx]
            if len(max_route) > 2:
                best_delta = float('inf')
                best_move = None
                for cust in max_route[1:-1]:
                    new_max_route = [x for x in max_route if x != cust]
                    new_max_len = route_length(new_max_route)
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other_route = current_routes[r_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                            delta = new_max_candidate - current_max
                            if delta < best_delta:
                                best_delta = delta
                                best_move = (cust, max_idx, r_idx, pos, new_max_candidate)
                if best_move is not None:
                    cust, from_idx, to_idx, pos, new_max_val = best_move
                    if best_delta < 0 or random.random() < exp(-best_delta / T):
                        current_routes[from_idx] = [x for x in current_routes[from_idx] if x != cust]
                        current_routes[to_idx].insert(pos, cust)
                        current_max = new_max_val
                        improved = True
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in current_routes]
                            report_best_vrp(best_routes)
        else:  # intra_2opt
            improved_local = False
            for r_idx in range(truck_count):
                route = current_routes[r_idx]
                if len(route) <= 3:
                    continue
                best_delta = 0.0
                best_ij = None
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        delta = route_length(new_route) - route_length(route)
                        if delta < best_delta:
                            best_delta = delta
                            best_ij = (i, k, r_idx)
                if best_ij:
                    i, k, r_idx = best_ij
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    old_len = route_length(route)
                    new_len = route_length(new_route)
                    other_lens = [route_length(current_routes[j]) for j in range(truck_count) if j != r_idx]
                    new_max_candidate = max(new_len, *other_lens)
                    delta = new_max_candidate - current_max
                    if delta < 0 or random.random() < exp(-delta / T):
                        current_routes[r_idx] = new_route
                        current_max = new_max_candidate
                        improved_local = True
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in current_routes]
                            report_best_vrp(best_routes)
            if improved_local:
                improved = True
        
        if improved:
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= 20:
                # Ruin-recreate perturbation
                route_lens = [(route_length(r), idx) for idx, r in enumerate(current_routes)]
                route_lens.sort(reverse=True)
                num_to_remove = max(1, (n-1)//10)
                removed = []
                for _, r_idx in route_lens:
                    route = current_routes[r_idx]
                    if len(route) <= 2:
                        continue
                    can_remove = min(num_to_remove - len(removed), len(route)-2)
                    if can_remove <= 0:
                        break
                    remove_set = set(random.sample(route[1:-1], can_remove))
                    for cust in remove_set:
                        removed.append((r_idx, cust))
                    current_routes[r_idx] = [x for x in route if x not in remove_set]
                    if len(removed) >= num_to_remove:
                        break
                unassigned = [cust for _, cust in removed]
                random.shuffle(unassigned)
                while unassigned:
                    best_cust = None
                    best_max_val = float('inf')
                    best_second_max = float('inf')
                    best_data = None
                    for cust in unassigned:
                        best_for_cust = None
                        second_for_cust = None
                        for r_idx, route in enumerate(current_routes):
                            for pos in range(1, len(route)):
                                prev = route[pos-1]
                                nxt = route[pos]
                                cost_inc = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                new_len = route_length(route) + cost_inc
                                other_lens = [route_length(current_routes[i]) for i in range(truck_count) if i != r_idx]
                                new_max_candidate = max(new_len, *other_lens)
                                if best_for_cust is None or new_max_candidate < best_for_cust[0]:
                                    second_for_cust = best_for_cust
                                    best_for_cust = (new_max_candidate, cost_inc, r_idx, pos)
                                elif second_for_cust is None or new_max_candidate < second_for_cust[0]:
                                    second_for_cust = (new_max_candidate, cost_inc, r_idx, pos)
                        if best_for_cust is not None:
                            regret = (second_for_cust[0] - best_for_cust[0]) if second_for_cust else 1e9
                            if best_cust is None or regret > (best_second_max - best_max_val):
                                best_cust = cust
                                best_max_val = best_for_cust[0]
                                best_second_max = second_for_cust[0] if second_for_cust else float('inf')
                                best_data = best_for_cust
                    if best_cust is None:
                        break
                    _, _, r_idx, pos = best_data
                    current_routes[r_idx].insert(pos, best_cust)
                    unassigned.remove(best_cust)
                current_max = max_route_len(current_routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in current_routes]
                    report_best_vrp(best_routes)
                stagnation = 0
    
    return best_routes
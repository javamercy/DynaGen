import numpy as np
import random
from math import exp

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def max_route_len(routes):
        return max(route_length(r) for r in routes) if routes else float('inf')
    
    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 10)
    
    for attempt in range(max_attempts):
        # Greedy construction: for each customer, insert into best position minimizing new max route distance
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            candidates = []
            for cust in unassigned:
                best_insert = None
                best_new_max = float('inf')
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_len = route_length(new_route)
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_insert = (r_idx, pos, new_max)
                candidates.append((best_new_max, best_insert[0], best_insert[1], cust))
            candidates.sort(key=lambda x: (x[0], x[3]))
            _, r_idx, pos, cust = candidates[0]
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        
        # Simulated annealing improvement
        T = current_max
        cooling_rate = 0.99
        max_iter = n * truck_count * 2
        stagnation = 0
        perturb_size = 0.10
        for it in range(max_iter):
            T *= cooling_rate
            if T < 1e-12:
                T = 1e-12
            improved = False
            # Inter-relocate: move a customer from longest route to another
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            route = routes[max_idx]
            if len(route) > 2:
                best_delta = 0
                best_move = None
                for cust in route[1:-1]:
                    new_max_route = [x for x in route if x != cust]
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
                            if new_max_candidate < current_max:
                                delta = current_max - new_max_candidate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (cust, max_idx, r_idx, pos, new_max_candidate)
                if best_move:
                    cust, from_idx, to_idx, pos, new_max_val = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    if new_max_val < current_max:
                        current_max = new_max_val
                        improved = True
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
                    else:
                        delta = new_max_val - current_max
                        if random.random() < exp(-delta / T):
                            current_max = new_max_val
                            improved = True
            # If no improvement from inter-relocate, try intra-2opt
            if not improved:
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
                            if new_len < route_length(route):
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
                stagnation = 0
                perturb_size = 0.10
            else:
                stagnation += 1
                if stagnation >= 10:
                    # Simple ruin-recreate: remove random customers from long routes and reinsert greedily
                    route_lengths = [(route_length(r), idx) for idx, r in enumerate(routes)]
                    route_lengths.sort(reverse=True)
                    num_to_remove = max(1, int((n-1) * perturb_size))
                    removed = []
                    for _, r_idx in route_lengths:
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
                        best_new_max = float('inf')
                        best_insert = None
                        for cust in unassigned:
                            for r_idx, route in enumerate(routes):
                                for pos in range(1, len(route)):
                                    new_route = route[:pos] + [cust] + route[pos:]
                                    new_len = route_length(new_route)
                                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                                    new_max = max(new_len, *other_lens)
                                    if new_max < best_new_max:
                                        best_new_max = new_max
                                        best_insert = (r_idx, pos)
                        if best_insert is None:
                            break
                        r_idx, pos = best_insert
                        routes[r_idx].insert(pos, best_cust)
                        unassigned.remove(best_cust)
                    current_max = max_route_len(routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                    perturb_size = min(perturb_size + 0.05, 0.25)
                    stagnation = 0
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
    if best_routes is None:
        best_routes = routes
    return best_routes
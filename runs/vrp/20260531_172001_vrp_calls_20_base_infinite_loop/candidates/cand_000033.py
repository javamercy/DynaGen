import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    # Step 1: Min-max greedy construction (from parent cand_000009)
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    while unassigned:
        best_customer = None
        best_route_idx = None
        best_pos = None
        best_max_after = float('inf')
        best_cost = None
        for cust in unassigned:
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(r) for i, r in enumerate(routes) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    # tie-break: smaller new_max, then larger cost, then smaller customer index
                    if (new_max < best_max_after or
                        (new_max == best_max_after and (best_cost is None or cost > best_cost)) or
                        (new_max == best_max_after and cost == best_cost and cust < best_customer)):
                        best_max_after = new_max
                        best_customer = cust
                        best_route_idx = r_idx
                        best_pos = pos
                        best_cost = cost
        routes[best_route_idx].insert(best_pos, best_customer)
        unassigned.remove(best_customer)
    
    # Step 2: Simulated Annealing
    current_solution = [r[:] for r in routes]
    current_max = max(route_length(r) for r in current_solution)
    best_solution = [r[:] for r in current_solution]
    best_max = current_max
    report_best_vrp(best_solution)
    
    T = current_max * 10
    T_min = 1e-6
    alpha = 0.99
    max_iter = n * truck_count * 50
    iter_count = 0
    
    while iter_count < max_iter and T > T_min:
        iter_count += 1
        move_type = random.choice(['relocate', 'swap', '2opt'])
        new_solution = None
        if move_type == 'relocate':
            non_empty = [i for i, r in enumerate(current_solution) if len(r) > 2]
            if not non_empty:
                continue
            from_idx = random.choice(non_empty)
            route_from = current_solution[from_idx]
            cust_idx = random.randrange(1, len(route_from)-1)
            cust = route_from[cust_idx]
            to_idx = random.randrange(truck_count)
            route_to = current_solution[to_idx]
            pos = random.randrange(1, len(route_to))
            new_solution = [r[:] for r in current_solution]
            new_solution[from_idx] = [x for x in new_solution[from_idx] if x != cust]
            new_solution[to_idx].insert(pos, cust)
        elif move_type == 'swap':
            non_empty = [i for i, r in enumerate(current_solution) if len(r) > 2]
            if len(non_empty) < 2:
                continue
            i = random.choice(non_empty)
            j = random.choice([x for x in non_empty if x != i])
            route_i = current_solution[i]
            route_j = current_solution[j]
            pos_i = random.randrange(1, len(route_i)-1)
            pos_j = random.randrange(1, len(route_j)-1)
            cust_i = route_i[pos_i]
            cust_j = route_j[pos_j]
            new_solution = [r[:] for r in current_solution]
            new_solution[i] = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
            new_solution[j] = route_j[:pos_j] + [cust_i] + route_j[pos_j+1:]
        else:  # 2opt
            eligible = [i for i, r in enumerate(current_solution) if len(r) >= 4]
            if not eligible:
                continue
            idx = random.choice(eligible)
            route = current_solution[idx]
            i = random.randrange(1, len(route)-2)
            k = random.randrange(i+1, len(route)-1)
            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
            new_solution = [r[:] for r in current_solution]
            new_solution[idx] = new_route
        
        if new_solution is None:
            continue
        
        new_max = max(route_length(r) for r in new_solution)
        if new_max <= current_max:
            accept = True
        else:
            delta = new_max - current_max
            if random.random() < math.exp(-delta / T):
                accept = True
            else:
                accept = False
        
        if accept:
            current_solution = new_solution
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_solution = [r[:] for r in current_solution]
                report_best_vrp(best_solution)
        T *= alpha
    
    return best_solution
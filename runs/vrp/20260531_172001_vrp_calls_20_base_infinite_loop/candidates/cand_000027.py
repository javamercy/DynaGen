import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def max_route_len(routes):
        return max(route_length(r) for r in routes)
    
    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 10)
    
    for attempt in range(max_attempts):
        # Construction with deterministic regret
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        while unassigned:
            best_regret = -float('inf')
            best_cust = None
            best_cost = float('inf')
            best_r_idx = None
            best_pos = None
            for cust in unassigned:
                costs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        costs.append((cost, r_idx, pos))
                costs.sort(key=lambda x: x[0])
                best_cost_cust = costs[0][0]
                second_cost_cust = costs[1][0] if len(costs) > 1 else best_cost_cust + 1e9
                regret = second_cost_cust - best_cost_cust
                if regret > best_regret or (regret == best_regret and cust < best_cust):
                    best_regret = regret
                    best_cust = cust
                    best_cost = best_cost_cust
                    best_r_idx = costs[0][1]
                    best_pos = costs[0][2]
            routes[best_r_idx].insert(best_pos, best_cust)
            unassigned.remove(best_cust)
        
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        
        # Simulated annealing parameters
        initial_T = current_max * 0.2
        final_T = 0.001
        cooling_rate = 0.99
        T = initial_T
        max_iter = n * truck_count * 2
        no_improve_count = 0
        perturbation_threshold = max(1, n // 3)
        
        for it in range(max_iter):
            improved = False
            # Inter-route relocate from longest route
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
                            if new_max_candidate < current_max:
                                delta = current_max - new_max_candidate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (cust, max_idx, r_idx, pos)
                if best_move:
                    cust, from_idx, to_idx, pos = best_move
                    new_routes = [r[:] for r in routes]
                    new_routes[from_idx] = [x for x in new_routes[from_idx] if x != cust]
                    new_routes[to_idx].insert(pos, cust)
                    new_max = max_route_len(new_routes)
                    # SA acceptance
                    if new_max < current_max or random.random() < math.exp(-(new_max - current_max) / T):
                        routes = new_routes
                        current_max = new_max
                        improved = True
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
            # Intra-route 2-opt (first improvement)
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
                            new_routes = [r[:] for r in routes]
                            new_routes[r_idx] = new_route
                            new_max = max_route_len(new_routes)
                            # SA acceptance
                            if new_max < current_max or random.random() < math.exp(-(new_max - current_max) / T):
                                routes = new_routes
                                current_max = new_max
                                improved = True
                                improved_intra = True
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                            break
                    if improved_intra:
                        break
            # Update temperature
            T = max(final_T, T * cooling_rate)
            # Perturbation if no improvement for a while
            if not improved:
                no_improve_count += 1
                if no_improve_count >= perturbation_threshold:
                    customers = list(range(1, n))
                    random.shuffle(customers)
                    num_perturb = max(1, n // 5)
                    for cust in customers[:num_perturb]:
                        for r_idx, route in enumerate(routes):
                            if cust in route:
                                routes[r_idx] = [x for x in route if x != cust]
                                break
                        r_idx = random.randrange(truck_count)
                        pos = random.randrange(1, len(routes[r_idx]))
                        routes[r_idx].insert(pos, cust)
                    current_max = max_route_len(routes)
                    no_improve_count = 0
            else:
                no_improve_count = 0
    
    if best_routes is None:
        best_routes = routes
    return best_routes
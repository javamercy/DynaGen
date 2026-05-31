import numpy as np
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
    
    def max_route_len(routes):
        return max(route_length(r) for r in routes)
    
    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 10)
    
    for attempt in range(max_attempts):
        # Construction: min-max regret with deterministic selection
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            candidates = []
            for cust in unassigned:
                insert_options = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_options.append((new_max, cost, r_idx, pos))
                insert_options.sort(key=lambda x: (x[0], x[1]))
                best = insert_options[0]
                second = insert_options[1] if len(insert_options) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            # sort by best_new_max asc, regret desc, cost desc, cust asc
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            # Deterministic selection: pick best candidate (highest regret among ties)
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
        lengths = [route_length(r) for r in routes]
        current_max = max(lengths)
        improved = True
        iter_count = 0
        max_iter = n * truck_count * 2
        stagnation = 0
        perturbation_threshold = max(1, n // 4)
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Inter-route relocate from longest route
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
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
                            if new_max_candidate < current_max - 1e-12:
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
                    stagnation = 0
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
                            if route_length(new_route) < route_length(route) - 1e-12:
                                route[:] = new_route
                                improved_intra = True
                                break
                        if improved_intra:
                            break
                    if improved_intra:
                        improved = True
                        stagnation = 0
                        new_max = max_route_len(routes)
                        if new_max < current_max - 1e-12:
                            current_max = new_max
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                        break
            
            if not improved:
                stagnation += 1
                if stagnation >= perturbation_threshold:
                    # Perturb: move 15% customers to random positions
                    customers = list(range(1, n))
                    random.shuffle(customers)
                    num_perturb = max(1, n * 15 // 100)
                    for cust in customers[:num_perturb]:
                        for r_idx, route in enumerate(routes):
                            if cust in route:
                                routes[r_idx] = [x for x in route if x != cust]
                                break
                        r_idx = random.randrange(truck_count)
                        pos = random.randrange(1, len(routes[r_idx]))
                        routes[r_idx].insert(pos, cust)
                    current_max = max_route_len(routes)
                    stagnation = 0
                    improved = True
        
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
    
    if best_routes is None:
        best_routes = routes
    return best_routes
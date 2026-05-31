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
        # Shuffle customers for diversity
        customers = list(range(1, n))
        random.shuffle(customers)
        
        # Initialize empty routes
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = customers[:]
        
        # Regret insertion
        while unassigned:
            best_insertions = []  # (regret, cust, best_cost, r_idx, pos)
            for cust in unassigned:
                costs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        costs.append((cost, r_idx, pos))
                costs.sort(key=lambda x: x[0])
                best_cost = costs[0][0]
                second_cost = costs[1][0] if len(costs) > 1 else best_cost + 1
                regret = second_cost - best_cost
                best_insertions.append((regret, cust, best_cost, costs[0][1], costs[0][2]))
            # Sort by regret descending, then by customer index ascending
            best_insertions.sort(key=lambda x: (-x[0], x[1]))
            regret, cust, best_cost, r_idx, pos = best_insertions[0]
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
        max_iter = n * truck_count * 2
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            
            # Inter-route relocate from longest route
            if len(max_route) > 2:
                best_move = None
                best_new_max = current_max
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
                            if new_max_candidate < best_new_max:
                                best_new_max = new_max_candidate
                                best_move = (cust, max_idx, r_idx, pos)
                if best_move:
                    cust, from_idx, to_idx, pos = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    current_max = best_new_max
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
            
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_len = route_length(new_route)
                        old_len = route_length(route)
                        if new_len < old_len:
                            route[:] = new_route
                            improved = True
                            new_max = max_route_len(routes)
                            if new_max < current_max:
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                            # break out of loops to restart
                            break
                    if improved:
                        break
        
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
    
    if best_routes is None:
        best_routes = routes
    return best_routes
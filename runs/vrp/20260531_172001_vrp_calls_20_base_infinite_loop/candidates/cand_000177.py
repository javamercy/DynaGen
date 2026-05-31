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
        return max(route_length(r) for r in routes) if routes else float('inf')
    
    best_routes = None
    best_max = float('inf')
    
    # Initialize pheromone matrix
    tau = np.ones((n, n)) * 0.01
    alpha = 1
    beta = 1
    rho = 0.1
    num_ants = max(1, min(truck_count * 2, 10))
    max_iter = max(5, min(n // 10, 20))
    
    for iteration in range(max_iter):
        for ant in range(num_ants):
            routes = [[0, 0] for _ in range(truck_count)]
            unassigned = set(range(1, n))
            while unassigned:
                candidates = []
                total_weight = 0.0
                for cust in unassigned:
                    for r_idx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            new_len = route_length(route) + distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                            other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                            new_max = max(new_len, *other_lens)
                            cost_increase = new_max - max_route_len(routes)
                            if cost_increase < 0:
                                cost_increase = 0.0
                            eta = 1.0 / (1.0 + cost_increase)
                            pheromone_avg = (tau[prev, cust] + tau[cust, nxt]) / 2.0
                            weight = (eta ** alpha) * (pheromone_avg ** beta)
                            if weight > 1e-12:
                                candidates.append((weight, cost_increase, r_idx, pos, cust))
                                total_weight += weight
                if total_weight == 0.0:
                    # Fallback to deterministic regret
                    fallback = []
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
                        if len(insert_info) > 1:
                            regret = insert_info[1][0] - insert_info[0][0]
                        else:
                            regret = 0.0
                        fallback.append((insert_info[0][0], regret, insert_info[0][1], insert_info[0][2], insert_info[0][3], cust))
                    fallback.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
                    chosen = fallback[0]
                    _, _, _, r_idx, pos, cust = chosen
                    routes[r_idx].insert(pos, cust)
                    unassigned.remove(cust)
                else:
                    r = random.random() * total_weight
                    cumulative = 0.0
                    chosen = None
                    for weight, _, r_idx, pos, cust in candidates:
                        cumulative += weight
                        if r <= cumulative:
                            chosen = (r_idx, pos, cust)
                            break
                    if chosen is None:
                        chosen = (candidates[-1][2], candidates[-1][3], candidates[-1][4])
                    r_idx, pos, cust = chosen
                    routes[r_idx].insert(pos, cust)
                    unassigned.remove(cust)
            
            current_max = max_route_len(routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(routes)
        
        # Pheromone update
        tau *= (1 - rho)
        if best_routes is not None:
            delta = 1.0 / (best_max + 1.0)
            for route in best_routes:
                for i in range(len(route)-1):
                    a = route[i]
                    b = route[i+1]
                    tau[a, b] += delta
                    tau[b, a] += delta
        tau = np.clip(tau, 0.001, None)
    
    if best_routes is None:
        # Fallback construction
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
                candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            chosen = candidates[0]
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        best_routes = routes
        best_max = max_route_len(routes)
        report_best_vrp(routes)
    
    return best_routes
import numpy as np
import math
from itertools import permutations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Special case: enough trucks to give each customer its own route
    if truck_count >= n - 1:
        routes = []
        for i in customers:
            routes.append([0, i, 0])
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes
    
    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    # Regret insertion construction
    unassigned = set(customers)
    while unassigned:
        best_customer = None
        best_regret = -1.0
        best_insertion_cost = None
        best_route_idx = None
        best_pos = None
        for cust in unassigned:
            costs = []  # (cost, route_idx, pos)
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    costs.append((cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            best_cost = costs[0][0]
            second_cost = costs[1][0] if len(costs) > 1 else best_cost + 1e9
            regret = second_cost - best_cost
            # Tie-breaking: larger best_cost, then smaller customer id
            if (regret > best_regret or
                (abs(regret - best_regret) < 1e-12 and (best_insertion_cost is None or best_cost > best_insertion_cost)) or
                (abs(regret - best_regret) < 1e-12 and abs(best_cost - best_insertion_cost) < 1e-12 and cust < best_customer)):
                best_regret = regret
                best_customer = cust
                best_insertion_cost = best_cost
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
        # Insert best customer
        routes[best_route_idx].insert(best_pos, best_customer)
        unassigned.remove(best_customer)
    
    # Initial best
    current_max = max(route_length(r) for r in routes)
    best_routes = [r[:] for r in routes]
    best_max = current_max
    report_best_vrp(routes)
    
    # Improvement loop
    max_iter = min(100, n * truck_count)
    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        lengths = [route_length(r) for r in routes]
        max_idx = max(range(truck_count), key=lambda i: lengths[i])
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            continue
        # Try moving each customer from max_route to other routes
        best_delta = 0.0
        best_move = None
        for cust in max_route[1:-1]:
            # Remove cust from max_route temporarily
            new_max_candidates = [x for x in max_route if x != cust]
            new_max_len = route_length(new_max_candidates)
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                other_route = routes[r_idx]
                # Find best insertion position in other_route
                best_increase = float('inf')
                best_pos = -1
                for pos in range(1, len(other_route)):
                    prev = other_route[pos-1]
                    nxt = other_route[pos]
                    inc = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if inc < best_increase:
                        best_increase = inc
                        best_pos = pos
                new_other_route = other_route[:best_pos] + [cust] + other_route[best_pos:]
                new_other_len = route_length(new_other_route)
                new_max_candidate = max(new_max_len, new_other_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)])
                if new_max_candidate < current_max - 1e-12:
                    delta = current_max - new_max_candidate
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (cust, max_idx, r_idx, best_pos)
        if best_move:
            cust, from_idx, to_idx, pos = best_move
            # Apply move
            routes[from_idx] = [x for x in routes[from_idx] if x != cust]
            routes[to_idx].insert(pos, cust)
            current_max = current_max - best_delta
            if current_max < best_max - 1e-12:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(routes)
            improved = True
        else:
            # Intra-route 2-opt on each route
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                improved_intra = True
                inner_iter = 0
                while improved_intra and inner_iter < len(route):
                    improved_intra = False
                    inner_iter += 1
                    for i in range(1, len(route)-2):
                        for k in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            if route_length(new_route) < route_length(route):
                                route[:] = new_route
                                improved_intra = True
                                new_max = max(route_length(r) for r in routes)
                                if new_max < current_max - 1e-12:
                                    current_max = new_max
                                    if current_max < best_max - 1e-12:
                                        best_max = current_max
                                        best_routes = [r[:] for r in routes]
                                        report_best_vrp(routes)
                                break
                        if improved_intra:
                            break
    return best_routes
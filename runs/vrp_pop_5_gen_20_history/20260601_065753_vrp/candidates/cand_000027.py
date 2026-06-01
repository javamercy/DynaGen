import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def compute_route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    # Greedy min-max insertion construction
    routes = [[0, 0] for _ in range(truck_count)]
    lengths = [0.0] * truck_count
    for cust in customers:
        best_max = float('inf')
        best_r = -1
        best_p = -1
        for r in range(truck_count):
            route = routes[r]
            for p in range(1, len(route)):
                prev = route[p-1]
                nxt = route[p]
                new_len = lengths[r] - distance_matrix[prev, nxt] + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                new_max = new_len
                for rr in range(truck_count):
                    if rr != r and lengths[rr] > new_max:
                        new_max = lengths[rr]
                if new_max < best_max or (new_max == best_max and (r < best_r or (r == best_r and p < best_p))):
                    best_max = new_max
                    best_r = r
                    best_p = p
        routes[best_r].insert(best_p, cust)
        lengths[best_r] = compute_route_length(routes[best_r])
    
    best_routes = [list(r) for r in routes]
    best_max = max(lengths)
    current_max = best_max
    
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
    
    report_best_vrp(routes)
    
    # Simulated annealing parameters
    max_iter = n * truck_count
    T0 = best_max * 0.1
    T_end = 0.001
    cooling_factor = (T_end / T0) ** (1.0 / max_iter) if max_iter > 0 else 1.0
    T = T0
    
    for iteration in range(max_iter):
        # Generate a random move
        move_type = random.randint(0, 2)
        best_new_max = float('inf')
        best_move = None
        best_tie = None
        
        if move_type == 0 or move_type == 1:  # relocate or swap
            # Pick two distinct routes with at least one customer each
            routes_with_cust = [r for r in range(truck_count) if len(routes[r]) > 2]
            if len(routes_with_cust) < 2:
                continue
            t1 = random.choice(routes_with_cust)
            # For relocate, t2 can be same as t1? Usually relocate is inter-route, but could be intra? We'll only do inter-route for simplicity.
            routes_with_possible_t2 = [r for r in range(truck_count) if r != t1]
            if not routes_with_possible_t2:
                continue
            t2 = random.choice(routes_with_possible_t2)
            idx1 = random.randint(1, len(routes[t1])-2)
            cust = routes[t1][idx1]
            
            if move_type == 0:  # relocate
                pos = random.randint(1, len(routes[t2])-1)
                # Evaluate move
                new_route1 = routes[t1][:idx1] + routes[t1][idx1+1:]
                new_route2 = routes[t2][:pos] + [cust] + routes[t2][pos:]
                len1_new = compute_route_length(new_route1)
                len2_new = compute_route_length(new_route2)
                new_max = max(len1_new, len2_new)
                for rr in range(truck_count):
                    if rr != t1 and rr != t2:
                        if lengths[rr] > new_max:
                            new_max = lengths[rr]
                tie = (new_max, t1, idx1, t2, pos)
                if best_move is None or tie < best_tie:
                    best_new_max = new_max
                    best_move = ('relocate', t1, idx1, t2, pos)
                    best_tie = tie
            else:  # swap
                if len(routes[t2]) <= 2:
                    continue
                idx2 = random.randint(1, len(routes[t2])-2)
                cust2 = routes[t2][idx2]
                new_route1 = routes[t1][:idx1] + [cust2] + routes[t1][idx1+1:]
                new_route2 = routes[t2][:idx2] + [cust] + routes[t2][idx2+1:]
                len1_new = compute_route_length(new_route1)
                len2_new = compute_route_length(new_route2)
                new_max = max(len1_new, len2_new)
                for rr in range(truck_count):
                    if rr != t1 and rr != t2:
                        if lengths[rr] > new_max:
                            new_max = lengths[rr]
                tie = (new_max, t1, idx1, t2, idx2)
                if best_move is None or tie < best_tie:
                    best_new_max = new_max
                    best_move = ('swap', t1, idx1, t2, idx2)
                    best_tie = tie
        else:  # 2-opt
            t = random.randrange(truck_count)
            if len(routes[t]) > 3:
                i = random.randint(1, len(routes[t])-3)
                j = random.randint(i+1, len(routes[t])-2)
                new_route = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
                new_len = compute_route_length(new_route)
                new_max = new_len
                for rr in range(truck_count):
                    if rr != t:
                        if lengths[rr] > new_max:
                            new_max = lengths[rr]
                tie = (new_max, t, i, j)
                if best_move is None or tie < best_tie:
                    best_new_max = new_max
                    best_move = ('2opt', t, i, j, new_route)
                    best_tie = tie
            else:
                continue
        
        if best_move is None:
            continue
        
        delta = best_new_max - current_max
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            if best_move[0] == 'relocate':
                _, t1, idx1, t2, pos = best_move
                cust = routes[t1][idx1]
                del routes[t1][idx1]
                routes[t2].insert(pos, cust)
                lengths[t1] = compute_route_length(routes[t1])
                lengths[t2] = compute_route_length(routes[t2])
            elif best_move[0] == 'swap':
                _, t1, idx1, t2, idx2 = best_move
                cust = routes[t1][idx1]
                cust2 = routes[t2][idx2]
                routes[t1][idx1] = cust2
                routes[t2][idx2] = cust
                lengths[t1] = compute_route_length(routes[t1])
                lengths[t2] = compute_route_length(routes[t2])
            else:
                _, t, i, j, new_route = best_move
                routes[t] = new_route
                lengths[t] = compute_route_length(new_route)
            current_max = max(lengths)
            report_best_vrp(routes)
        
        T *= cooling_factor
    
    return best_routes
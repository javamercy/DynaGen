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
    
    # Tabu search parameters
    max_iter = n * truck_count * 5
    tabu_tenure = max(5, n // 10)
    tabu_list = []
    num_candidates = min(20, n)
    
    for iteration in range(max_iter):
        candidates = []
        # Generate candidate moves
        for _ in range(num_candidates):
            move_type = random.randint(0, 2)
            if move_type == 0:  # relocate
                routes_with_cust = [r for r in range(truck_count) if len(routes[r]) > 2]
                if len(routes_with_cust) < 2:
                    continue
                t1 = random.choice(routes_with_cust)
                t2 = random.choice([r for r in range(truck_count) if r != t1 and len(routes[r]) >= 2])
                if t2 is None or t1 == t2:
                    continue
                idx1 = random.randint(1, len(routes[t1])-2)
                pos = random.randint(1, len(routes[t2])-1)
                cust = routes[t1][idx1]
                new_route1 = routes[t1][:idx1] + routes[t1][idx1+1:]
                new_route2 = routes[t2][:pos] + [cust] + routes[t2][pos:]
                new_len1 = compute_route_length(new_route1)
                new_len2 = compute_route_length(new_route2)
                new_max = max(new_len1, new_len2)
                for rr in range(truck_count):
                    if rr != t1 and rr != t2:
                        if lengths[rr] > new_max:
                            new_max = lengths[rr]
                tie = (new_max, 0, t1, idx1, t2, pos)
                candidates.append((tie, ('relocate', t1, idx1, t2, pos, new_route1, new_route2)))
            elif move_type == 1:  # swap
                routes_with_cust = [r for r in range(truck_count) if len(routes[r]) > 2]
                if len(routes_with_cust) < 2:
                    continue
                t1 = random.choice(routes_with_cust)
                t2 = random.choice([r for r in range(truck_count) if r != t1 and len(routes[r]) > 2])
                if t2 is None:
                    continue
                idx1 = random.randint(1, len(routes[t1])-2)
                idx2 = random.randint(1, len(routes[t2])-2)
                cust = routes[t1][idx1]
                cust2 = routes[t2][idx2]
                new_route1 = routes[t1][:idx1] + [cust2] + routes[t1][idx1+1:]
                new_route2 = routes[t2][:idx2] + [cust] + routes[t2][idx2+1:]
                new_len1 = compute_route_length(new_route1)
                new_len2 = compute_route_length(new_route2)
                new_max = max(new_len1, new_len2)
                for rr in range(truck_count):
                    if rr != t1 and rr != t2:
                        if lengths[rr] > new_max:
                            new_max = lengths[rr]
                tie = (new_max, 1, t1, idx1, t2, idx2)
                candidates.append((tie, ('swap', t1, idx1, t2, idx2, new_route1, new_route2)))
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
                    tie = (new_max, 2, t, i, j)
                    candidates.append((tie, ('2opt', t, i, j, new_route)))
        if not candidates:
            continue
        # Sort candidates by tie tuple (new_max, type, indices)
        candidates.sort(key=lambda x: x[0])
        best_move = None
        best_tie = None
        for tie, move in candidates:
            if move[0] == '2opt':
                move_key = (2, move[1], move[2], move[3])
            elif move[0] == 'relocate':
                move_key = (0, move[1], move[2], move[3], move[4])
            else:  # swap
                move_key = (1, move[1], move[2], move[3], move[4])
            is_tabu = any(mk == move_key for mk in tabu_list)
            if is_tabu:
                if tie[0] < best_max:  # aspiration
                    best_move = move
                    best_tie = tie
                    break
                else:
                    continue
            else:
                best_move = move
                best_tie = tie
                break
        if best_move is None:
            best_tie, best_move = candidates[0]
        # Apply move
        if best_move[0] == 'relocate':
            _, t1, idx1, t2, pos, new_r1, new_r2 = best_move
            routes[t1] = new_r1
            routes[t2] = new_r2
            lengths[t1] = compute_route_length(routes[t1])
            lengths[t2] = compute_route_length(routes[t2])
            tabu_key = (0, t1, idx1, t2, pos)
        elif best_move[0] == 'swap':
            _, t1, idx1, t2, idx2, new_r1, new_r2 = best_move
            routes[t1] = new_r1
            routes[t2] = new_r2
            lengths[t1] = compute_route_length(routes[t1])
            lengths[t2] = compute_route_length(routes[t2])
            tabu_key = (1, t1, idx1, t2, idx2)
        else:  # 2opt
            _, t, i, j, new_route = best_move
            routes[t] = new_route
            lengths[t] = compute_route_length(new_route)
            tabu_key = (2, t, i, j)
        current_max = max(lengths)
        report_best_vrp(routes)
        # Update tabu list
        tabu_list.append(tabu_key)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
    
    return best_routes
import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    # Construction: regret-based min-max insertion
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = set(customers)
    
    while unassigned:
        best_regret = -1.0
        selected_cust = None
        best_info = None  # (t, pos)
        
        for cust in list(unassigned):
            best_new_max = float('inf')
            second_best_new_max = float('inf')
            best_pos = None
            best_t = None
            for t in range(truck_count):
                route = routes[t]
                # positions from 1 to len(route)-1 (depot at ends)
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_len = route_distance(new_route)
                    new_max = new_len
                    for k in range(truck_count):
                        if k != t:
                            new_max = max(new_max, route_dists[k])
                    if new_max < best_new_max:
                        second_best_new_max = best_new_max
                        best_new_max = new_max
                        best_pos = pos
                        best_t = t
                    elif new_max < second_best_new_max:
                        second_best_new_max = new_max
            regret = second_best_new_max - best_new_max
            # tie-break: larger regret; if equal, smaller customer index
            if regret > best_regret or (regret == best_regret and cust < selected_cust):
                best_regret = regret
                selected_cust = cust
                best_info = (best_t, best_pos)
        
        # Insert selected customer
        if selected_cust is None:
            # fallback: just insert in first route at first position
            selected_cust = min(unassigned)
            best_info = (0, 1)
        t, pos = best_info
        routes[t] = routes[t][:pos] + [selected_cust] + routes[t][pos:]
        route_dists[t] = route_distance(routes[t])
        unassigned.remove(selected_cust)
    
    # Initial best solution
    best_max = max(route_dists)
    best_routes = [list(r) for r in routes]
    
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(route_distance(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
    
    report_best_vrp(routes)
    
    # Tabu search parameters
    tabu_tenure = 7
    tabu_list = []
    tabu_set = set()
    max_iter = (n - 1) * truck_count * 2
    
    for _ in range(max_iter):
        best_move = None
        best_new_max = float('inf')
        best_tie = None
        
        # Relocate moves
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust = route1[idx1]
                new_route1 = route1[:idx1] + route1[idx1+1:]
                dist1_new = route_distance(new_route1)
                for t2 in range(truck_count):
                    if t1 == t2:
                        continue
                    route2 = routes[t2]
                    for idx2 in range(1, len(route2)):
                        new_route2 = route2[:idx2] + [cust] + route2[idx2:]
                        dist2_new = route_distance(new_route2)
                        new_max = max(dist1_new, dist2_new)
                        for k in range(truck_count):
                            if k != t1 and k != t2:
                                new_max = max(new_max, route_dists[k])
                        # tabu check: reverse move (cust, t2, t1) is tabu?
                        is_tabu = (cust, t2, t1) in tabu_set
                        # aspiration: if new_max < best_max, ignore tabu
                        if is_tabu and new_max >= best_max:
                            continue
                        # tie-break: (new_max, move_type=0, t1, idx1, t2, idx2)
                        tie = (new_max, 0, t1, idx1, t2, idx2)
                        if tie < best_tie if best_tie is not None else True:
                            best_new_max = new_max
                            best_move = ('relocate', t1, idx1, t2, idx2, cust)
                            best_tie = tie
        
        # Swap moves
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust1 = route1[idx1]
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        cust2 = route2[idx2]
                        new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                        new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                        dist1_new = route_distance(new_route1)
                        dist2_new = route_distance(new_route2)
                        new_max = max(dist1_new, dist2_new)
                        for k in range(truck_count):
                            if k != t1 and k != t2:
                                new_max = max(new_max, route_dists[k])
                        # tabu check: two reverse entries
                        is_tabu = ((cust1, cust2, t1, t2) in tabu_set) or ((cust2, cust1, t2, t1) in tabu_set)
                        if is_tabu and new_max >= best_max:
                            continue
                        tie = (new_max, 1, t1, idx1, t2, idx2)
                        if tie < best_tie if best_tie is not None else True:
                            best_new_max = new_max
                            best_move = ('swap', t1, idx1, t2, idx2, cust1, cust2)
                            best_tie = tie
        
        # 2-opt moves
        for t in range(truck_count):
            route = routes[t]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    new_max = new_dist
                    for k in range(truck_count):
                        if k != t:
                            new_max = max(new_max, route_dists[k])
                    # 2-opt moves not tabu
                    tie = (new_max, 2, t, i, j, -1, -1)
                    if tie < best_tie if best_tie is not None else True:
                        best_new_max = new_max
                        best_move = ('2opt', t, i, j, new_route)
                        best_tie = tie
        
        if best_move is None:
            break
        
        # Apply best move
        if best_move[0] == 'relocate':
            _, t1, idx1, t2, idx2, cust = best_move
            routes[t1] = routes[t1][:idx1] + routes[t1][idx1+1:]
            routes[t2] = routes[t2][:idx2] + [cust] + routes[t2][idx2:]
            route_dists[t1] = route_distance(routes[t1])
            route_dists[t2] = route_distance(routes[t2])
            # Add reverse move to tabu
            tabu_entry = (cust, t2, t1)
            tabu_list.append(tabu_entry)
            tabu_set.add(tabu_entry)
        elif best_move[0] == 'swap':
            _, t1, idx1, t2, idx2, cust1, cust2 = best_move
            routes[t1][idx1] = cust2
            routes[t2][idx2] = cust1
            route_dists[t1] = route_distance(routes[t1])
            route_dists[t2] = route_distance(routes[t2])
            tabu_entry1 = (cust1, cust2, t1, t2)
            tabu_entry2 = (cust2, cust1, t2, t1)
            tabu_list.append(tabu_entry1)
            tabu_set.add(tabu_entry1)
            tabu_list.append(tabu_entry2)
            tabu_set.add(tabu_entry2)
        else:  # '2opt'
            _, t, i, j, new_route = best_move
            routes[t] = new_route
            route_dists[t] = route_distance(new_route)
        
        # Manage tabu list length
        while len(tabu_list) > tabu_tenure:
            entry = tabu_list.pop(0)
            tabu_set.discard(entry)
        
        report_best_vrp(routes)
    
    return best_routes
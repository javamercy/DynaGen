import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    # Construction: start with each customer as a route, then merge until we have truck_count routes
    routes = [[0, i, 0] for i in range(1, n)]
    if len(routes) < truck_count:
        # add empty routes
        routes += [[0, 0] for _ in range(truck_count - len(routes))]
    elif len(routes) > truck_count:
        # merge until we have exactly truck_count routes
        while len(routes) > truck_count:
            best_new_max = float('inf')
            best_pair = None
            best_orientation = None
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    r1 = routes[i]
                    r2 = routes[j]
                    dist1 = route_distance(r1)
                    dist2 = route_distance(r2)
                    # orientation 1: r1 then r2
                    last1 = r1[-2]
                    first2 = r2[1]
                    new_len = dist1 + dist2 + distance_matrix[last1, first2]
                    new_max = new_len
                    for k, r in enumerate(routes):
                        if k != i and k != j:
                            new_max = max(new_max, route_distance(r))
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_pair = (i, j)
                        best_orientation = 1
                    # orientation 2: r2 then r1
                    last2 = r2[-2]
                    first1 = r1[1]
                    new_len = dist1 + dist2 + distance_matrix[last2, first1]
                    new_max = new_len
                    for k, r in enumerate(routes):
                        if k != i and k != j:
                            new_max = max(new_max, route_distance(r))
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_pair = (i, j)
                        best_orientation = 2
            # Apply best merge
            i, j = best_pair
            if best_orientation == 1:
                new_route = routes[i][:-1] + routes[j][1:]
            else:
                new_route = routes[j][:-1] + routes[i][1:]
            routes[i] = new_route
            routes.pop(j)
    
    # Compute current distances
    route_dists = [route_distance(r) for r in routes]
    best_max = max(route_dists)
    best_routes = [list(r) for r in routes]
    
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(route_distance(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
    
    report_best_vrp(routes)
    
    # Tabu search with adaptive tenure
    tabu_tenure = max(5, int(math.sqrt(n)))
    tabu_list = []
    tabu_set = set()
    max_iter = (n - 1) * truck_count * 2
    no_improve_iter = 0
    
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
                        # Tabu check: reverse move (cust, t2, t1) is tabu?
                        is_tabu = (cust, t2, t1) in tabu_set
                        new_tie = (new_max, 0, t1, idx1, t2, idx2)  # 0 for relocate
                        if (not is_tabu or new_max < best_max) and (new_max < best_new_max or (new_max == best_new_max and new_tie < best_tie)):
                            best_new_max = new_max
                            best_move = ('relocate', t1, idx1, t2, idx2, cust)
                            best_tie = new_tie
        
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
                        # Tabu check: store swap as (cust1, cust2, t1, t2) and (cust2, cust1, t2, t1)
                        is_tabu = ((cust1, cust2, t1, t2) in tabu_set) or ((cust2, cust1, t2, t1) in tabu_set)
                        new_tie = (new_max, 1, t1, idx1, t2, idx2)  # 1 for swap
                        if (not is_tabu or new_max < best_max) and (new_max < best_new_max or (new_max == best_new_max and new_tie < best_tie)):
                            best_new_max = new_max
                            best_move = ('swap', t1, idx1, t2, idx2, cust1, cust2)
                            best_tie = new_tie
        
        # 2-opt moves (intra-route, no tabu to keep simple)
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
                    new_tie = (new_max, 2, t, i, j)
                    if new_max < best_new_max or (new_max == best_new_max and new_tie < best_tie):
                        best_new_max = new_max
                        best_move = ('2opt', t, i, j, new_route)
                        best_tie = new_tie
        
        if best_move is None or best_new_max >= best_max:
            # If no improvement, adjust tenure and continue? Break to avoid infinite loop, but we have finite max_iter
            break
        
        # Apply best move
        entries_to_add = []
        if best_move[0] == 'relocate':
            _, t1, idx1, t2, idx2, cust = best_move
            # Remove cust from t1
            routes[t1] = routes[t1][:idx1] + routes[t1][idx1+1:]
            # Insert into t2
            routes[t2] = routes[t2][:idx2] + [cust] + routes[t2][idx2:]
            route_dists[t1] = route_distance(routes[t1])
            route_dists[t2] = route_distance(routes[t2])
            # Add reverse move to tabu
            tabu_entry = (cust, t2, t1)
            entries_to_add.append(tabu_entry)
        elif best_move[0] == 'swap':
            _, t1, idx1, t2, idx2, cust1, cust2 = best_move
            routes[t1][idx1] = cust2
            routes[t2][idx2] = cust1
            route_dists[t1] = route_distance(routes[t1])
            route_dists[t2] = route_distance(routes[t2])
            tabu_entry1 = (cust1, cust2, t1, t2)
            tabu_entry2 = (cust2, cust1, t2, t1)
            entries_to_add.append(tabu_entry1)
            entries_to_add.append(tabu_entry2)
        else:  # '2opt'
            _, t, i, j, new_route = best_move
            routes[t] = new_route
            route_dists[t] = route_distance(new_route)
            # 2-opt does not add tabu entries
        
        for entry in entries_to_add:
            tabu_list.append(entry)
            tabu_set.add(entry)
        
        # Manage tabu list length: remove oldest entries until length <= tenure
        while len(tabu_list) > tabu_tenure:
            entry = tabu_list.pop(0)
            tabu_set.discard(entry)
            # If entry is a swap pair, we might have removed only one; but it's okay
        
        current_max = max(route_dists)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(routes)
            no_improve_iter = 0
            # Increase tenure slightly on improvement
            tabu_tenure = min(20, tabu_tenure + 1)
        else:
            no_improve_iter += 1
            # Decrease tenure if no improvement for a while
            if no_improve_iter >= 10:
                tabu_tenure = max(5, tabu_tenure - 1)
                no_improve_iter = 0
    
    return best_routes
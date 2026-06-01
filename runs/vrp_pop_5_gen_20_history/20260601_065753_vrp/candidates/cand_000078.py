import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def total_distance(route):
        return route_distance(route)
    
    best_max = float('inf')
    best_routes = None
    
    def update_best(routes):
        nonlocal best_max, best_routes
        m = max(route_distance(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    
    # Construction: greedy best insertion minimizing max route distance
    # Deterministic tie-breaking by total distance (lower total distance preferred)
    def construct():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            candidates = []
            for cust in list(unassigned):
                best_inc = float('inf')
                best_truck = -1
                best_pos = -1
                best_total_inc = float('inf')
                for t in range(truck_count):
                    route = routes[t]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_distance(new_route)
                        new_max = new_dist
                        for k in range(truck_count):
                            if k != t:
                                new_max = max(new_max, route_dists[k])
                        total_inc = new_dist - route_dists[t]
                        if new_max < best_inc or (new_max == best_inc and total_inc < best_total_inc):
                            best_inc = new_max
                            best_truck = t
                            best_pos = pos
                            best_total_inc = total_inc
                candidates.append((best_inc, best_total_inc, cust, best_truck, best_pos))
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            _, _, cust, t, pos = candidates[0]
            routes[t] = routes[t][:pos] + [cust] + routes[t][pos:]
            route_dists[t] = route_distance(routes[t])
            unassigned.remove(cust)
        return routes, route_dists
    
    routes, route_dists = construct()
    update_best(routes)
    
    # Tabu search parameters
    tabu_tenure_min = 3
    tabu_tenure_max = 12
    current_tenure = 7
    tabu_list = []
    tabu_set = set()
    max_iter = (n - 1) * truck_count * 2
    iterations_since_improvement = 0
    improvement_threshold = max(5, n // 5)
    
    # Perturbation settings (modified: ruin and recreate)
    perturb_threshold = max_iter // 4
    max_perturb = 3
    perturb_count = 0
    
    for _ in range(max_iter):
        best_move = None
        best_new_max = float('inf')
        best_tie = None
        # Relocate
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for i in range(1, len(route1)-1):
                cust = route1[i]
                new_route1 = route1[:i] + route1[i+1:]
                dist1_new = route_distance(new_route1)
                for t2 in range(truck_count):
                    if t1 == t2:
                        continue
                    route2 = routes[t2]
                    for j in range(1, len(route2)):
                        new_route2 = route2[:j] + [cust] + route2[j:]
                        dist2_new = route_distance(new_route2)
                        new_max = max(dist1_new, dist2_new)
                        for k in range(truck_count):
                            if k != t1 and k != t2:
                                new_max = max(new_max, route_dists[k])
                        is_tabu = (cust, t2, t1) in tabu_set
                        if is_tabu and new_max >= best_max:
                            continue
                        tie = (new_max, 0, t1, i, t2, j)
                        if best_tie is None or tie < best_tie:
                            best_new_max = new_max
                            best_move = ('relocate', t1, i, t2, j, cust)
                            best_tie = tie
        # Swap
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for i in range(1, len(route1)-1):
                cust1 = route1[i]
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for j in range(1, len(route2)-1):
                        cust2 = route2[j]
                        new_route1 = route1[:i] + [cust2] + route1[i+1:]
                        new_route2 = route2[:j] + [cust1] + route2[j+1:]
                        dist1_new = route_distance(new_route1)
                        dist2_new = route_distance(new_route2)
                        new_max = max(dist1_new, dist2_new)
                        for k in range(truck_count):
                            if k != t1 and k != t2:
                                new_max = max(new_max, route_dists[k])
                        is_tabu = ((cust1, cust2, t1, t2) in tabu_set) or ((cust2, cust1, t2, t1) in tabu_set)
                        if is_tabu and new_max >= best_max:
                            continue
                        tie = (new_max, 1, t1, i, t2, j)
                        if best_tie is None or tie < best_tie:
                            best_new_max = new_max
                            best_move = ('swap', t1, i, t2, j, cust1, cust2)
                            best_tie = tie
        # 2-opt
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
                    tie = (new_max, 2, t, i, j)
                    if best_tie is None or tie < best_tie:
                        best_new_max = new_max
                        best_move = ('2opt', t, i, j, new_route)
                        best_tie = tie
        
        if best_move is None:
            break
        
        # Apply move
        improved = False
        if best_move[0] == 'relocate':
            _, t1, i, t2, j, cust = best_move
            routes[t1] = routes[t1][:i] + routes[t1][i+1:]
            routes[t2] = routes[t2][:j] + [cust] + routes[t2][j:]
            route_dists[t1] = route_distance(routes[t1])
            route_dists[t2] = route_distance(routes[t2])
            tabu_entry = (cust, t2, t1)
            tabu_list.append(tabu_entry)
            tabu_set.add(tabu_entry)
        elif best_move[0] == 'swap':
            _, t1, i, t2, j, cust1, cust2 = best_move
            routes[t1][i] = cust2
            routes[t2][j] = cust1
            route_dists[t1] = route_distance(routes[t1])
            route_dists[t2] = route_distance(routes[t2])
            tabu_entry1 = (cust1, cust2, t1, t2)
            tabu_entry2 = (cust2, cust1, t2, t1)
            tabu_list.append(tabu_entry1)
            tabu_set.add(tabu_entry1)
            tabu_list.append(tabu_entry2)
            tabu_set.add(tabu_entry2)
        else:  # 2opt
            _, t, i, j, new_route = best_move
            routes[t] = new_route
            route_dists[t] = route_distance(new_route)
        
        # Manage tabu list
        while len(tabu_list) > current_tenure:
            entry = tabu_list.pop(0)
            tabu_set.discard(entry)
        
        old_best_max = best_max
        update_best(routes)
        if best_max < old_best_max:
            improved = True
            iterations_since_improvement = 0
            if current_tenure > tabu_tenure_min:
                current_tenure -= 1
        else:
            iterations_since_improvement += 1
            if iterations_since_improvement >= improvement_threshold and current_tenure < tabu_tenure_max:
                current_tenure += 1
                iterations_since_improvement = 0
        
        # Perturbation if stalled (ruin and recreate)
        if iterations_since_improvement >= perturb_threshold and perturb_count < max_perturb:
            # Identify the route with the maximum distance
            max_route_idx = np.argmax(route_dists)
            max_route = routes[max_route_idx]
            # Remove a portion of customers from the longest route (e.g., 20% of all customers)
            num_remove = max(1, int(0.2 * (n - 1)))
            # Also consider removing some from other routes to balance
            to_remove = []
            # First, from the longest route, remove interior customers
            interior_cust = max_route[1:-1]
            if len(interior_cust) > 0:
                # Remove a fraction
                remove_from_long = min(len(interior_cust), max(1, num_remove // 2))
                # Deterministically remove the first `remove_from_long` customers (always the same order)
                to_remove.extend(interior_cust[:remove_from_long])
            # Also remove some from other routes if needed
            remaining = num_remove - len(to_remove)
            if remaining > 0:
                other_routes = [r for idx, r in enumerate(routes) if idx != max_route_idx]
                # Collect all interior customers from other routes
                other_custs = []
                for r in other_routes:
                    other_custs.extend(r[1:-1])
                # Deterministically take the first `remaining`
                to_remove.extend(other_custs[:remaining])
            # Remove those customers from their routes
            for cust in to_remove:
                for t in range(truck_count):
                    if cust in routes[t]:
                        idx = routes[t].index(cust)
                        if idx != 0 and idx != len(routes[t])-1:
                            routes[t] = routes[t][:idx] + routes[t][idx+1:]
                            route_dists[t] = route_distance(routes[t])
                        break
            # Reinsert removed customers using greedy insertion that minimizes max distance
            unassigned_perturb = set(to_remove)
            while unassigned_perturb:
                candidates = []
                for cust in list(unassigned_perturb):
                    best_inc = float('inf')
                    best_truck = -1
                    best_pos = -1
                    best_total_inc = float('inf')
                    for t in range(truck_count):
                        route = routes[t]
                        for pos in range(1, len(route)):
                            new_route = route[:pos] + [cust] + route[pos:]
                            new_dist = route_distance(new_route)
                            new_max = new_dist
                            for k in range(truck_count):
                                if k != t:
                                    new_max = max(new_max, route_dists[k])
                            total_inc = new_dist - route_dists[t]
                            if new_max < best_inc or (new_max == best_inc and total_inc < best_total_inc):
                                best_inc = new_max
                                best_truck = t
                                best_pos = pos
                                best_total_inc = total_inc
                    candidates.append((best_inc, best_total_inc, cust, best_truck, best_pos))
                candidates.sort(key=lambda x: (x[0], x[1], x[2]))
                _, _, cust, t, pos = candidates[0]
                routes[t] = routes[t][:pos] + [cust] + routes[t][pos:]
                route_dists[t] = route_distance(routes[t])
                unassigned_perturb.remove(cust)
            # Reset tabu list
            tabu_list.clear()
            tabu_set.clear()
            current_tenure = 7
            iterations_since_improvement = 0
            perturb_count += 1
            update_best(routes)
    
    return best_routes
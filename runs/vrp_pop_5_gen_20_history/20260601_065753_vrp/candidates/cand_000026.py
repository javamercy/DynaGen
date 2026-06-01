import numpy as np
import random

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
    
    # Construction: regret-based min-max insertion
    def construct_solution():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            best_regret = -1.0
            selected_cust = None
            best_info = None
            for cust in list(unassigned):
                best_new_max = float('inf')
                second_best_new_max = float('inf')
                best_t = -1
                best_pos = -1
                for t in range(truck_count):
                    route = routes[t]
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
                            best_t = t
                            best_pos = pos
                        elif new_max < second_best_new_max:
                            second_best_new_max = new_max
                regret = second_best_new_max - best_new_max
                if regret > best_regret or (regret == best_regret and cust < selected_cust):
                    best_regret = regret
                    selected_cust = cust
                    best_info = (best_t, best_pos)
            t, pos = best_info
            routes[t] = routes[t][:pos] + [selected_cust] + routes[t][pos:]
            route_dists[t] = route_distance(routes[t])
            unassigned.remove(selected_cust)
        return routes, max(route_dists)
    
    # Initial solution
    best_routes, best_max = construct_solution()
    current_routes = [list(r) for r in best_routes]
    current_max = best_max
    
    def evaluate(routes):
        return max(route_distance(r) for r in routes)
    
    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        m = evaluate(routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
    
    report_best_vrp(current_routes)
    
    # Local search: best-improvement (single pass)
    def local_search(routes, max_len):
        improved = True
        iteration = 0
        max_iter = 1000
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            new_routes = None
            new_max = max_len
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
                            new_max_candidate = max(dist1_new, dist2_new)
                            for k in range(truck_count):
                                if k != t1 and k != t2:
                                    new_max_candidate = max(new_max_candidate, route_distance(routes[k]))
                            if new_max_candidate < new_max:
                                new_max = new_max_candidate
                                new_routes = [list(r) for r in routes]
                                new_routes[t1] = new_route1
                                new_routes[t2] = new_route2
                                improved = True
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
                            new_max_candidate = max(dist1_new, dist2_new)
                            for k in range(truck_count):
                                if k != t1 and k != t2:
                                    new_max_candidate = max(new_max_candidate, route_distance(routes[k]))
                            if new_max_candidate < new_max:
                                new_max = new_max_candidate
                                new_routes = [list(r) for r in routes]
                                new_routes[t1] = new_route1
                                new_routes[t2] = new_route2
                                improved = True
            # 2-opt
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        new_max_candidate = new_dist
                        for k in range(truck_count):
                            if k != t:
                                new_max_candidate = max(new_max_candidate, route_distance(routes[k]))
                        if new_max_candidate < new_max:
                            new_max = new_max_candidate
                            new_routes = [list(r) for r in routes]
                            new_routes[t] = new_route
                            improved = True
            if improved:
                routes = new_routes
                max_len = new_max
        return routes, max_len
    
    # VNS parameters
    k_max = 3
    iter_max = 50 * n
    for outer_iter in range(iter_max):
        # Shaking: apply k random moves
        k = (outer_iter % k_max) + 1
        shake_routes = [list(r) for r in current_routes]
        for _ in range(k):
            move_type = random.choice(['relocate', 'swap', '2opt'])
            if move_type == 'relocate':
                # Find a feasible relocate move
                feasible = False
                attempts = 0
                while not feasible and attempts < 100:
                    t1 = random.randrange(truck_count)
                    route1 = shake_routes[t1]
                    if len(route1) > 2:
                        i = random.randrange(1, len(route1)-1)
                        cust = route1[i]
                        t2 = random.randrange(truck_count)
                        if t2 != t1:
                            route2 = shake_routes[t2]
                            j = random.randrange(1, len(route2))
                            new_shake = [list(r) for r in shake_routes]
                            new_shake[t1].pop(i)
                            new_shake[t2].insert(j, cust)
                            shake_routes = new_shake
                            feasible = True
                    attempts += 1
            elif move_type == 'swap':
                feasible = False
                attempts = 0
                while not feasible and attempts < 100:
                    t1 = random.randrange(truck_count)
                    route1 = shake_routes[t1]
                    if len(route1) > 2:
                        i = random.randrange(1, len(route1)-1)
                        cust1 = route1[i]
                        t2 = random.randrange(truck_count)
                        if t2 != t1:
                            route2 = shake_routes[t2]
                            if len(route2) > 2:
                                j = random.randrange(1, len(route2)-1)
                                cust2 = route2[j]
                                new_shake = [list(r) for r in shake_routes]
                                new_shake[t1][i] = cust2
                                new_shake[t2][j] = cust1
                                shake_routes = new_shake
                                feasible = True
                    attempts += 1
            else: #2opt
                feasible = False
                attempts = 0
                while not feasible and attempts < 100:
                    t = random.randrange(truck_count)
                    route = shake_routes[t]
                    if len(route) > 3:
                        i = random.randrange(1, len(route)-2)
                        j = random.randrange(i+1, len(route)-1)
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_shake = [list(r) for r in shake_routes]
                        new_shake[t] = new_route
                        shake_routes = new_shake
                        feasible = True
                    attempts += 1
        # Local search on shaken solution
        shake_max = evaluate(shake_routes)
        improved_routes, improved_max = local_search(shake_routes, shake_max)
        report_best_vrp(improved_routes)
        # Acceptance: only better or equal (deterministic)
        if improved_max <= current_max:
            current_routes = improved_routes
            current_max = improved_max
        # If not better, keep current, shaking continues
    return best_routes
import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def total_distance(routes):
        return sum(route_distance(r) for r in routes)
    
    def copy_routes(routes):
        return [r[:] for r in routes]
    
    global_best_routes = None
    global_best_max = float('inf')
    restarts = max(10, n // 5)
    
    for restart in range(restarts):
        # Construction: randomized greedy insertion with random tie-breaking
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        
        for cust in customers:
            best_increase = float('inf')
            best_route = -1
            best_pos = -1
            candidates = []
            current_max = max_distance(routes)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_distance(route) + added
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    candidates.append((increase, random.random(), r_idx, pos))
            # sort by increase, then random tie-break
            candidates.sort(key=lambda x: (x[0], x[1]))
            best_candidate = candidates[0]
            routes[best_candidate[2]].insert(best_candidate[3], cust)
        
        best_routes = copy_routes(routes)
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        
        # Adaptive parameters
        shake_intensity = 1.0  # probability of shake after each cycle
        no_improve_shake = 0
        max_no_improve_shake = 3
        
        for cycle in range(5):  # cycles of local search and shake
            # Local search phases on longest routes
            max_iter = (n - 1) * truck_count * 10
            no_improve_count = 0
            for iteration in range(max_iter):
                improved = False
                max_dist = max_distance(routes)
                longest_routes = [i for i, r in enumerate(routes) if route_distance(r) == max_dist]
                phases = ['2opt', 'relocate', 'swap', 'cross']
                random.shuffle(phases)
                for phase in phases:
                    if improved:
                        break
                    if phase == '2opt':
                        for r_idx in longest_routes:
                            route = routes[r_idx]
                            if len(route) <= 3:
                                continue
                            for i in range(1, len(route)-2):
                                for j in range(i+1, len(route)-1):
                                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                    old_dist = route_distance(route)
                                    new_dist = route_distance(new_route)
                                    if new_dist >= old_dist:
                                        continue
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx)
                                    new_max = max(new_dist, other_max)
                                    if new_max < best_max - 1e-12:
                                        routes[r_idx] = new_route
                                        best_routes = copy_routes(routes)
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    elif phase == 'relocate':
                        for src in longest_routes:
                            route_src = routes[src]
                            if len(route_src) <= 2:
                                continue
                            for pos_src in range(1, len(route_src)-1):
                                cust = route_src[pos_src]
                                temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                                dist_src = route_distance(temp_src)
                                for dst in range(truck_count):
                                    if dst == src:
                                        continue
                                    route_dst = routes[dst]
                                    for pos_dst in range(1, len(route_dst)):
                                        new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                                        dist_dst = route_distance(new_dst)
                                        other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst)
                                        new_max = max(dist_src, dist_dst, other_max)
                                        if new_max < best_max - 1e-12:
                                            routes[src] = temp_src
                                            routes[dst] = new_dst
                                            best_routes = copy_routes(routes)
                                            best_max = new_max
                                            improved = True
                                            report_best_vrp(best_routes)
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    elif phase == 'swap':
                        for t1 in longest_routes:
                            route1 = routes[t1]
                            if len(route1) <= 2:
                                continue
                            for t2 in range(truck_count):
                                if t2 == t1:
                                    continue
                                route2 = routes[t2]
                                if len(route2) <= 2:
                                    continue
                                for i in range(1, len(route1)-1):
                                    for j in range(1, len(route2)-1):
                                        cust1 = route1[i]
                                        cust2 = route2[j]
                                        new_route1 = route1[:i] + [cust2] + route1[i+1:]
                                        new_route2 = route2[:j] + [cust1] + route2[j+1:]
                                        dist1 = route_distance(new_route1)
                                        dist2 = route_distance(new_route2)
                                        other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                        new_max = max(dist1, dist2, other_max)
                                        if new_max < best_max - 1e-12:
                                            routes[t1] = new_route1
                                            routes[t2] = new_route2
                                            best_routes = copy_routes(routes)
                                            best_max = new_max
                                            improved = True
                                            report_best_vrp(best_routes)
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    elif phase == 'cross':
                        for t1 in longest_routes:
                            route1 = routes[t1]
                            if len(route1) <= 2:
                                continue
                            for t2 in range(truck_count):
                                if t2 == t1:
                                    continue
                                route2 = routes[t2]
                                if len(route2) <= 2:
                                    continue
                                for i in range(1, len(route1)-1):
                                    for j in range(1, len(route2)-1):
                                        new_route1 = route1[:i] + route2[j:]
                                        new_route2 = route2[:j] + route1[i:]
                                        dist1 = route_distance(new_route1)
                                        dist2 = route_distance(new_route2)
                                        other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                        new_max = max(dist1, dist2, other_max)
                                        if new_max < best_max - 1e-12:
                                            routes[t1] = new_route1
                                            routes[t2] = new_route2
                                            best_routes = copy_routes(routes)
                                            best_max = new_max
                                            improved = True
                                            report_best_vrp(best_routes)
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                if improved:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= 5:
                        break
            
            # Shake: remove longest route and reinsert, if stalling also second-longest
            if random.random() < shake_intensity or no_improve_shake >= max_no_improve_shake:
                # Perturb the longest route
                longest_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]))
                longest_route = routes[longest_idx]
                if len(longest_route) > 2:
                    removed = longest_route[1:-1]
                    routes[longest_idx] = [0, 0]
                    for cust in removed:
                        best_increase = float('inf')
                        best_route = -1
                        best_pos = -1
                        candidates = []
                        current_max = max_distance(routes)
                        for r_idx in range(truck_count):
                            route = routes[r_idx]
                            for pos in range(1, len(route)):
                                prev = route[pos-1]
                                nxt = route[pos]
                                added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                new_route_dist = route_distance(route) + added
                                other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                                new_max = max(new_route_dist, other_max)
                                increase = new_max - current_max
                                candidates.append((increase, random.random(), r_idx, pos))
                        candidates.sort(key=lambda x: (x[0], x[1]))
                        best_candidate = candidates[0]
                        routes[best_candidate[2]].insert(best_candidate[3], cust)
                    cur_max = max_distance(routes)
                    if cur_max < best_max - 1e-12:
                        best_routes = copy_routes(routes)
                        best_max = cur_max
                        report_best_vrp(best_routes)
                        no_improve_shake = 0
                    else:
                        no_improve_shake += 1
                else:
                    no_improve_shake += 1
                
                # If stalling, also perturb the second-longest route
                if no_improve_shake >= max_no_improve_shake:
                    sorted_indices = sorted(range(truck_count), key=lambda i: route_distance(routes[i]), reverse=True)
                    if len(sorted_indices) >= 2:
                        second_idx = sorted_indices[1]
                        second_route = routes[second_idx]
                        if len(second_route) > 2:
                            removed2 = second_route[1:-1]
                            routes[second_idx] = [0, 0]
                            for cust in removed2:
                                best_increase = float('inf')
                                best_route = -1
                                best_pos = -1
                                candidates = []
                                current_max = max_distance(routes)
                                for r_idx in range(truck_count):
                                    route = routes[r_idx]
                                    for pos in range(1, len(route)):
                                        prev = route[pos-1]
                                        nxt = route[pos]
                                        added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                        new_route_dist = route_distance(route) + added
                                        other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                                        new_max = max(new_route_dist, other_max)
                                        increase = new_max - current_max
                                        candidates.append((increase, random.random(), r_idx, pos))
                                candidates.sort(key=lambda x: (x[0], x[1]))
                                best_candidate = candidates[0]
                                routes[best_candidate[2]].insert(best_candidate[3], cust)
                            cur_max = max_distance(routes)
                            if cur_max < best_max - 1e-12:
                                best_routes = copy_routes(routes)
                                best_max = cur_max
                                report_best_vrp(best_routes)
                                no_improve_shake = 0
                            else:
                                no_improve_shake = 0  # reset anyway
                        else:
                            no_improve_shake = 0
                # Adjust intensity adaptively
                if no_improve_shake == 0:
                    shake_intensity = max(0.2, shake_intensity * 0.9)
                else:
                    shake_intensity = min(1.0, shake_intensity * 1.1)
        
        # Global best update
        if best_max < global_best_max - 1e-12:
            global_best_max = best_max
            global_best_routes = copy_routes(best_routes)
            report_best_vrp(global_best_routes)
    
    if global_best_routes is None:
        global_best_routes = copy_routes(routes)
    return global_best_routes
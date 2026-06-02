import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    global_best_routes = None
    global_best_max = float('inf')
    
    restarts = max(5, n // 10)
    for _ in range(restarts):
        # Regret-2 construction with GRASP
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        
        while customers:
            # Compute regret-2 for each unassigned customer
            best_inc = [float('inf')] * len(customers)
            second_inc = [float('inf')] * len(customers)
            best_route_idx = [-1] * len(customers)
            best_pos_idx = [-1] * len(customers)
            for idx, cust in enumerate(customers):
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        inc = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        current_dist = route_distance(route)
                        new_dist = current_dist + inc
                        other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                        new_max = max(new_dist, other_max)
                        if new_max < best_inc[idx]:
                            second_inc[idx] = best_inc[idx]
                            best_inc[idx] = new_max
                            best_route_idx[idx] = r_idx
                            best_pos_idx[idx] = pos
                        elif new_max < second_inc[idx]:
                            second_inc[idx] = new_max
            # Compute regret = second_best - best
            regrets = [second_inc[i] - best_inc[i] for i in range(len(customers))]
            # Build RCL: top k (k = max(1, len(customers)//3))
            k = max(1, len(customers) // 3)
            sorted_indices = sorted(range(len(customers)), key=lambda i: regrets[i], reverse=True)
            rcl = sorted_indices[:k]
            chosen_idx = random.choice(rcl)
            cust = customers[chosen_idx]
            r_idx = best_route_idx[chosen_idx]
            pos = best_pos_idx[chosen_idx]
            routes[r_idx].insert(pos, cust)
            customers.pop(chosen_idx)
        
        # Initial solution
        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        
        # Local search
        max_iter = (n - 1) * truck_count * 5
        no_improve_count = 0
        for _ in range(max_iter):
            improved = False
            phases = ['2opt', 'relocate', 'swap', 'cross']
            random.shuffle(phases)
            for phase in phases:
                if phase == '2opt':
                    for r_idx in range(truck_count):
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
                                if new_max < best_max:
                                    routes[r_idx] = new_route
                                    best_routes = [r[:] for r in routes]
                                    best_max = new_max
                                    improved = True
                                    report_best_vrp(best_routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'relocate':
                    for src in range(truck_count):
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
                                    if new_max < best_max:
                                        routes[src] = temp_src
                                        routes[dst] = new_dst
                                        best_routes = [r[:] for r in routes]
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
                    for t1 in range(truck_count):
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
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
                                    if new_max < best_max:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
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
                    for t1 in range(truck_count):
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
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
                                    if new_max < best_max:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
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
                    break
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= 5:
                    break
        
        # Update global best
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [r[:] for r in best_routes]
            report_best_vrp(global_best_routes)
        if global_best_routes is None:
            global_best_routes = [r[:] for r in best_routes]
            global_best_max = best_max
            report_best_vrp(global_best_routes)
    
    return global_best_routes
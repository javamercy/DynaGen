import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def repair_routes(routes):
        for i, r in enumerate(routes):
            if r[0] != 0:
                routes[i] = [0] + r
            if r[-1] != 0:
                routes[i] = routes[i] + [0]
        return routes
    
    def copy_routes(routes):
        return [r[:] for r in routes]
    
    global_best_routes = None
    global_best_max = float('inf')
    restarts = max(5, n // 10)
    
    for restart in range(restarts):
        # Construction: cheapest insertion balancing max route distance
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_increase = float('inf')
            best_route = None
            best_pos = None
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
                    if increase < best_increase:
                        best_increase = increase
                        best_route = r_idx
                        best_pos = pos
                    # tie-breaking: choose first encountered (deterministic)
            routes[best_route].insert(best_pos, cust)
        
        best_routes = copy_routes(routes)
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        
        # Iterated Local Search with VND
        max_iter = (n - 1) * truck_count * 5
        no_improve = 0
        for iteration in range(max_iter):
            improved = False
            phases = ['2opt', 'relocate', 'swap', '2optstar']
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
                elif phase == '2optstar':
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
                    break
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 3:
                    break
        
        # Perturbation for next restart: block moves
        if restart < restarts - 1:
            perturbed = copy_routes(best_routes)
            for _ in range(random.randint(1, 3)):
                src = random.randint(0, truck_count-1)
                if len(perturbed[src]) <= 3:
                    continue
                length = random.randint(1, min(3, len(perturbed[src])-2))
                start = random.randint(1, len(perturbed[src])-length-1)
                block = perturbed[src][start:start+length]
                perturbed[src] = perturbed[src][:start] + perturbed[src][start+length:]
                dst = random.randint(0, truck_count-1)
                pos = random.randint(1, len(perturbed[dst])-1)
                perturbed[dst] = perturbed[dst][:pos] + block + perturbed[dst][pos:]
            routes = perturbed
        
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = copy_routes(best_routes)
    
    if global_best_routes is None:
        global_best_routes = [[0,0] for _ in range(truck_count)]
    routes = repair_routes(global_best_routes)
    return routes
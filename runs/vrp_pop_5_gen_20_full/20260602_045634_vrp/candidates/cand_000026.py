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
    
    def compute_routes(assignment):
        # assignment: list of customer truck indices, returns routes
        routes = [[0,0] for _ in range(truck_count)]
        for cust, t in enumerate(assignment):
            routes[t].insert(-1, cust+1)
        return routes
    
    best_routes = None
    best_max = float('inf')
    restarts = max(5, n // 10)
    restarts = min(restarts, 20)  # cap for speed
    
    no_improve_restarts = 0
    
    for restart in range(restarts):
        # Construction: random insertion greedy
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0,0] for _ in range(truck_count)]
        for cust in customers:
            best_cost = float('inf')
            best_truck = -1
            best_pos = -1
            for t in range(truck_count):
                route = routes[t]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    increase = distance_matrix[prev,cust] + distance_matrix[cust,nxt] - distance_matrix[prev,nxt]
                    if increase < best_cost:
                        best_cost = increase
                        best_truck = t
                        best_pos = pos
            routes[best_truck].insert(best_pos, cust)
        
        initial_routes = [r[:] for r in routes]
        initial_max = max_distance(initial_routes)
        if initial_max < best_max:
            best_routes = [r[:] for r in initial_routes]
            best_max = initial_max
            report_best_vrp(best_routes)
        
        # VND
        current_routes = [r[:] for r in initial_routes]
        current_max = initial_max
        improved = True
        max_vnd_iter = 50  # bounded
        for vnd_iter in range(max_vnd_iter):
            if not improved:
                break
            improved = False
            neighborhoods = ['2opt', 'relocate', 'swap', 'cross']
            random.shuffle(neighborhoods)
            for neigh in neighborhoods:
                if neigh == '2opt':
                    for t in range(truck_count):
                        route = current_routes[t]
                        if len(route) <= 3:
                            continue
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                old_dist = route_distance(route)
                                new_dist = route_distance(new_route)
                                if new_dist >= old_dist:
                                    continue
                                other_routes = [current_routes[x][:] for x in range(truck_count) if x != t]
                                other_max = max(route_distance(r) for r in other_routes) if other_routes else 0
                                new_max = max(new_dist, other_max)
                                if new_max < best_max:
                                    current_routes[t] = new_route
                                    best_routes = [r[:] for r in current_routes]
                                    best_max = new_max
                                    improved = True
                                    report_best_vrp(best_routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif neigh == 'relocate':
                    for src in range(truck_count):
                        route_src = current_routes[src]
                        if len(route_src) <= 2:
                            continue
                        for pos_src in range(1, len(route_src)-1):
                            cust = route_src[pos_src]
                            temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                            dist_src = route_distance(temp_src)
                            for dst in range(truck_count):
                                if dst == src:
                                    continue
                                route_dst = current_routes[dst]
                                for pos_dst in range(1, len(route_dst)):
                                    new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                                    dist_dst = route_distance(new_dst)
                                    other_routes = [current_routes[x][:] for x in range(truck_count) if x not in (src, dst)]
                                    other_max = max(route_distance(r) for r in other_routes) if other_routes else 0
                                    new_max = max(dist_src, dist_dst, other_max)
                                    if new_max < best_max:
                                        current_routes[src] = temp_src
                                        current_routes[dst] = new_dst
                                        best_routes = [r[:] for r in current_routes]
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
                elif neigh == 'swap':
                    for t1 in range(truck_count):
                        route1 = current_routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            route2 = current_routes[t2]
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
                                    other_routes = [current_routes[x][:] for x in range(truck_count) if x not in (t1, t2)]
                                    other_max = max(route_distance(r) for r in other_routes) if other_routes else 0
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max:
                                        current_routes[t1] = new_route1
                                        current_routes[t2] = new_route2
                                        best_routes = [r[:] for r in current_routes]
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
                elif neigh == 'cross':
                    for t1 in range(truck_count):
                        route1 = current_routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            route2 = current_routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    new_route1 = route1[:i] + route2[j:]
                                    new_route2 = route2[:j] + route1[i:]
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_routes = [current_routes[x][:] for x in range(truck_count) if x not in (t1, t2)]
                                    other_max = max(route_distance(r) for r in other_routes) if other_routes else 0
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max:
                                        current_routes[t1] = new_route1
                                        current_routes[t2] = new_route2
                                        best_routes = [r[:] for r in current_routes]
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
                current_max = best_max
        
        # Check improvement over global best
        if best_max == initial_max:
            no_improve_restarts += 1
        else:
            no_improve_restarts = 0
        
        if no_improve_restarts >= 5:
            break
    
    return best_routes
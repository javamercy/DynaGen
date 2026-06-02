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
        # Regret-2 construction with random tie-breaking
        routes = [[0,0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        random.shuffle(unassigned)  # initial shuffle for diversity
        while unassigned:
            best_inc = {}
            second_inc = {}
            best_pos = {}
            for cust in unassigned:
                best = float('inf')
                second = float('inf')
                bpos = None
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        inc = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if inc < best:
                            second = best
                            best = inc
                            bpos = (r_idx, pos)
                        elif inc < second:
                            second = inc
                best_inc[cust] = best
                second_inc[cust] = second
                best_pos[cust] = bpos
            regrets = {c: second_inc[c] - best_inc[c] for c in unassigned}
            max_reg = max(regrets.values())
            candidates = [c for c in unassigned if regrets[c] == max_reg]
            if len(candidates) > 1:
                min_best = min(best_inc[c] for c in candidates)
                candidates = [c for c in candidates if best_inc[c] == min_best]
            chosen = random.choice(candidates)
            r_idx, pos = best_pos[chosen]
            routes[r_idx].insert(pos, chosen)
            unassigned.remove(chosen)
        
        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        
        max_iter = (n - 1) * truck_count * 5
        no_improve_count = 0
        for _ in range(max_iter):
            improved = False
            phases = ['relocate', 'swap', '2opt', 'cross']
            random.shuffle(phases)
            for phase in phases:
                if phase == 'relocate':
                    longest_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]))
                    src = longest_idx
                    route_src = routes[src]
                    if len(route_src) > 2:
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
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x not in (src, dst))
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
                elif phase == 'swap':
                    longest_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]))
                    for other in range(truck_count):
                        if other == longest_idx or len(routes[other]) <= 2:
                            continue
                        route1 = routes[longest_idx]
                        route2 = routes[other]
                        for i in range(1, len(route1)-1):
                            for j in range(1, len(route2)-1):
                                cust1 = route1[i]
                                cust2 = route2[j]
                                new_route1 = route1[:i] + [cust2] + route1[i+1:]
                                new_route2 = route2[:j] + [cust1] + route2[j+1:]
                                dist1 = route_distance(new_route1)
                                dist2 = route_distance(new_route2)
                                other_max = max(route_distance(routes[x]) for x in range(truck_count) if x not in (longest_idx, other))
                                new_max = max(dist1, dist2, other_max)
                                if new_max < best_max:
                                    routes[longest_idx] = new_route1
                                    routes[other] = new_route2
                                    best_routes = [r[:] for r in routes]
                                    best_max = new_max
                                    improved = True
                                    report_best_vrp(best_routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == '2opt':
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                new_dist = route_distance(new_route)
                                if new_dist >= route_distance(route):
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
                elif phase == 'cross':
                    longest_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]))
                    for other in range(truck_count):
                        if other == longest_idx or len(routes[other]) <= 2:
                            continue
                        route1 = routes[longest_idx]
                        route2 = routes[other]
                        for i in range(1, len(route1)-1):
                            for j in range(1, len(route2)-1):
                                new_route1 = route1[:i] + route2[j:]
                                new_route2 = route2[:j] + route1[i:]
                                dist1 = route_distance(new_route1)
                                dist2 = route_distance(new_route2)
                                other_max = max(route_distance(routes[x]) for x in range(truck_count) if x not in (longest_idx, other))
                                new_max = max(dist1, dist2, other_max)
                                if new_max < best_max:
                                    routes[longest_idx] = new_route1
                                    routes[other] = new_route2
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
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= 3:
                    break
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [r[:] for r in best_routes]
            report_best_vrp(global_best_routes)
    if global_best_routes is None:
        global_best_routes = [r[:] for r in best_routes]
        report_best_vrp(global_best_routes)
    return global_best_routes
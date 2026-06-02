import numpy as np
import random
import math
from collections import defaultdict

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)
    
    def repair(routes):
        for i, r in enumerate(routes):
            if r[0] != 0:
                routes[i] = [0] + r
            if r[-1] != 0:
                routes[i] = routes[i] + [0]
        return routes
    
    # Regret-2 construction
    def construct():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        while customers:
            best_cust = None
            best_regret = -float('inf')
            best_route = None
            best_pos = None
            best_new_max = float('inf')
            for cust in customers:
                incs = []
                for r_idx, route in enumerate(routes):
                    current_max = max_route_dist(routes)
                    best_inc_local = float('inf')
                    best_pos_local = None
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_dist(new_route)
                        other_max = max(route_dist(routes[i]) for i in range(truck_count) if i != r_idx)
                        new_max = max(new_dist, other_max)
                        inc = new_max - current_max
                        if inc < best_inc_local - 1e-12:
                            best_inc_local = inc
                            best_pos_local = pos
                            best_new_max_local = new_max
                if best_pos_local is not None:
                    incs.append((best_inc_local, r_idx, best_pos_local, best_new_max_local))
                if not incs:
                    continue
                incs.sort(key=lambda x: x[0])
                best_inc, best_r, best_p, best_new = incs[0]
                if len(incs) > 1:
                    second_inc = incs[1][0]
                else:
                    second_inc = float('inf')
                regret = second_inc - best_inc
                # deterministic tie-breaking: smallest customer index
                if regret > best_regret + 1e-12:
                    best_regret = regret
                    best_cust = cust
                    best_route = best_r
                    best_pos = best_p
                    best_new_max = best_new
                elif abs(regret - best_regret) < 1e-12 and best_cust is not None and cust < best_cust:
                    best_regret = regret
                    best_cust = cust
                    best_route = best_r
                    best_pos = best_p
                    best_new_max = best_new
            if best_cust is None:
                # fallback: insert at first feasible position
                for cust in customers:
                    for r_idx, route in enumerate(routes):
                        pos = 1
                        routes[r_idx].insert(pos, cust)
                        best_cust = cust
                        break
                    if best_cust:
                        break
            else:
                routes[best_route].insert(best_pos, best_cust)
            customers.remove(best_cust)
        return routes
    
    # Local search with multiple neighborhoods
    def local_search(routes):
        best_routes = [r[:] for r in routes]
        best_max = max_route_dist(routes)
        improved = True
        max_iter = (n-1) * truck_count * 10
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # Define neighborhoods in random order for diversity but deterministic scanning within
            neighborhoods = ['2opt', 'relocate', 'swap', 'cross', '2optstar', 'oropt']
            random.shuffle(neighborhoods)
            for phase in neighborhoods:
                if phase == '2opt':
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                old_dist = route_dist(route)
                                new_dist = route_dist(new_route)
                                if new_dist >= old_dist:
                                    continue
                                other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != r_idx)
                                new_max = max(new_dist, other_max)
                                if new_max < best_max - 1e-12:
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
                    # inter-route relocate
                    for src in range(truck_count):
                        route_src = routes[src]
                        if len(route_src) <= 2:
                            continue
                        for pos_src in range(1, len(route_src)-1):
                            cust = route_src[pos_src]
                            temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                            dist_src = route_dist(temp_src)
                            for dst in range(truck_count):
                                if dst == src:
                                    continue
                                route_dst = routes[dst]
                                for pos_dst in range(1, len(route_dst)):
                                    new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                                    dist_dst = route_dist(new_dst)
                                    other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != src and x != dst)
                                    new_max = max(dist_src, dist_dst, other_max)
                                    if new_max < best_max - 1e-12:
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
                                    dist1 = route_dist(new_route1)
                                    dist2 = route_dist(new_route2)
                                    other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
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
                                    if new_route1[0] != 0:
                                        new_route1 = [0] + new_route1
                                    if new_route1[-1] != 0:
                                        new_route1.append(0)
                                    if new_route2[0] != 0:
                                        new_route2 = [0] + new_route2
                                    if new_route2[-1] != 0:
                                        new_route2.append(0)
                                    dist1 = route_dist(new_route1)
                                    dist2 = route_dist(new_route2)
                                    other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
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
                                    # ensure depot at start and end
                                    if new_route1[0] != 0:
                                        new_route1 = [0] + new_route1
                                    if new_route1[-1] != 0:
                                        new_route1.append(0)
                                    if new_route2[0] != 0:
                                        new_route2 = [0] + new_route2
                                    if new_route2[-1] != 0:
                                        new_route2.append(0)
                                    dist1 = route_dist(new_route1)
                                    dist2 = route_dist(new_route2)
                                    other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max - 1e-12:
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
                elif phase == 'oropt':
                    # Or-opt: move a sequence of length l=1,2,3 to another position within the same route
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        for l in range(1, min(4, len(route)-2)):
                            for i in range(1, len(route)-l-1):
                                seq = route[i:i+l]
                                remaining = route[:i] + route[i+l:]
                                for j in range(1, len(remaining)-1):
                                    if j == i:
                                        continue
                                    new_route = remaining[:j] + seq + remaining[j:]
                                    new_dist = route_dist(new_route)
                                    other_max = max(route_dist(routes[x]) for x in range(truck_count) if x != r_idx)
                                    new_max = max(new_dist, other_max)
                                    if new_max < best_max - 1e-12:
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
                        if improved:
                            break
                if improved:
                    break
        return best_routes
    
    # Perturbation: remove a block of customers from a random route and insert into another route
    def perturb(routes, intensity=1):
        for _ in range(intensity):
            # pick source route with at least 3 customers (depots not counted)
            feasible = [i for i, r in enumerate(routes) if len(r) > 3]
            if not feasible:
                continue
            src = random.choice(feasible)
            route = routes[src]
            max_block_len = min(3, len(route)-2)
            block_len = random.randint(1, max_block_len)
            start = random.randint(1, len(route)-1-block_len)
            block = route[start:start+block_len]
            del route[start:start+block_len]
            # insert block into random destination route and position
            dst = random.randint(0, truck_count-1)
            pos = random.randint(1, max(1, len(routes[dst])-1))
            for c in block:
                routes[dst].insert(pos, c)
                pos += 1
        return routes
    
    global_best_routes = None
    global_best_max = float('inf')
    restarts = max(10, n // 5)
    no_improve_restarts = 0
    for restart in range(restarts):
        routes = construct()
        routes = local_search(routes)
        current_max = max_route_dist(routes)
        if current_max < global_best_max - 1e-12:
            global_best_max = current_max
            global_best_routes = [r[:] for r in routes]
            no_improve_restarts = 0
        else:
            no_improve_restarts += 1
        intensity = 1 + (no_improve_restarts // 3)
        if restart < restarts - 1:
            # perturb the current best solution or the current routes
            if random.random() < 0.5 and global_best_routes is not None:
                base = [r[:] for r in global_best_routes]
            else:
                base = [r[:] for r in routes]
            routes = perturb(base, intensity)
    
    if global_best_routes is None:
        global_best_routes = [[0, 0] for _ in range(truck_count)]
    routes = repair(global_best_routes)
    return routes
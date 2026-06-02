import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def repair(routes):
        for i, r in enumerate(routes):
            if r[0] != 0:
                routes[i] = [0] + r
            if r[-1] != 0:
                routes[i] = routes[i] + [0]
        return routes
    
    # Regret-2 construction minimizing max route distance
    def construct():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_increase = []
            current_max = max_route_distance(routes)
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
                    best_increase.append((increase, r_idx, pos))
            best_increase.sort(key=lambda x: x[0])  # ascending increase
            # Regret-2: choose the insertion that has the largest difference between best and second best increases
            if len(best_increase) >= 2:
                best_diff = -float('inf')
                best_choice = best_increase[0]
                # Group by route? Simpler: compute regret for each candidate route
                # We'll compute regret per route: difference between best and second best increase for that route
                # But best_increase is sorted overall. Instead, we compute regret as second_min - min for each route
                # Alternative: simple regret: for each route, find best increase, then pick route with max regret
                route_best = {}
                for inc, r_idx, pos in best_increase:
                    if r_idx not in route_best:
                        route_best[r_idx] = (inc, pos)
                # For routes with at least 2 candidates, compute regret
                regrets = []
                for r_idx in range(truck_count):
                    best_inc = None
                    second_inc = None
                    for inc, ridx, pos in best_increase:
                        if ridx == r_idx:
                            if best_inc is None:
                                best_inc = (inc, pos)
                            else:
                                if inc < best_inc[0]:
                                    second_inc = best_inc
                                    best_inc = (inc, pos)
                                elif second_inc is None or inc < second_inc[0]:
                                    second_inc = (inc, pos)
                    if best_inc is not None:
                        if second_inc is not None:
                            regret = second_inc[0] - best_inc[0]
                        else:
                            regret = best_inc[0]  # or 0? treat as 0
                        regrets.append((regret, r_idx, best_inc[1]))
                if regrets:
                    regrets.sort(key=lambda x: -x[0])
                    # tie-break randomly
                    best_regret = regrets[0][0]
                    candidates = [r for r in regrets if abs(r[0] - best_regret) < 1e-12]
                    chosen = random.choice(candidates)
                    best_route, best_pos = chosen[1], chosen[2]
                else:
                    # fallback to first best
                    best_route, best_pos = best_increase[0][1], best_increase[0][2]
            else:
                best_route, best_pos = best_increase[0][1], best_increase[0][2]
            routes[best_route].insert(best_pos, cust)
        return routes
    
    # Local search phases
    def local_search(routes):
        best_routes = [r[:] for r in routes]
        best_max = max_route_distance(routes)
        improved = True
        max_iter = (n - 1) * truck_count * 2
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            phases = ['2opt', 'relocate', 'swap', 'cross', '2optstar']
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
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
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
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
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
                                    # Ensure depot start/end
                                    if new_route1[0] != 0:
                                        new_route1 = [0] + new_route1
                                    if new_route1[-1] != 0:
                                        new_route1.append(0)
                                    if new_route2[0] != 0:
                                        new_route2 = [0] + new_route2
                                    if new_route2[-1] != 0:
                                        new_route2.append(0)
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
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
                if improved:
                    break
        return best_routes
    
    # Perturbation: remove random customers and reinsert with regret
    def perturb(routes, intensity=1):
        removed = []
        for _ in range(intensity * min(5, max(1, (n-1)//10))):
            # Choose random route with at least one customer
            r_idx = random.randint(0, truck_count-1)
            while len(routes[r_idx]) <= 2:
                r_idx = random.randint(0, truck_count-1)
            pos = random.randint(1, len(routes[r_idx])-2)
            removed.append(routes[r_idx].pop(pos))
        # Reinsert removed customers using regret-2 construction
        random.shuffle(removed)
        for cust in removed:
            best_increase = []
            current_max = max_route_distance(routes)
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
                    best_increase.append((increase, r_idx, pos))
            best_increase.sort(key=lambda x: x[0])
            if len(best_increase) >= 2:
                route_best = {}
                for inc, r_idx, pos in best_increase:
                    if r_idx not in route_best:
                        route_best[r_idx] = (inc, pos)
                regrets = []
                for r_idx in range(truck_count):
                    best_inc = None
                    second_inc = None
                    for inc, ridx, pos in best_increase:
                        if ridx == r_idx:
                            if best_inc is None:
                                best_inc = (inc, pos)
                            else:
                                if inc < best_inc[0]:
                                    second_inc = best_inc
                                    best_inc = (inc, pos)
                                elif second_inc is None or inc < second_inc[0]:
                                    second_inc = (inc, pos)
                    if best_inc is not None:
                        if second_inc is not None:
                            regret = second_inc[0] - best_inc[0]
                        else:
                            regret = best_inc[0]
                        regrets.append((regret, r_idx, best_inc[1]))
                if regrets:
                    regrets.sort(key=lambda x: -x[0])
                    best_regret = regrets[0][0]
                    candidates = [r for r in regrets if abs(r[0] - best_regret) < 1e-12]
                    chosen = random.choice(candidates)
                    best_route, best_pos = chosen[1], chosen[2]
                else:
                    best_route, best_pos = best_increase[0][1], best_increase[0][2]
            else:
                best_route, best_pos = best_increase[0][1], best_increase[0][2]
            routes[best_route].insert(best_pos, cust)
        return routes
    
    global_best_routes = None
    global_best_max = float('inf')
    restarts = max(10, n // 5)
    no_improve_restarts = 0
    for restart in range(restarts):
        routes = construct()
        routes = local_search(routes)
        current_max = max_route_distance(routes)
        if current_max < global_best_max - 1e-12:
            global_best_max = current_max
            global_best_routes = [r[:] for r in routes]
            no_improve_restarts = 0
        else:
            no_improve_restarts += 1
        intensity = 1 + (no_improve_restarts // 2)
        if restart < restarts - 1:
            if random.random() < 0.5:
                base = [r[:] for r in global_best_routes]
            else:
                base = [r[:] for r in routes]
            routes = perturb(base, intensity)
    
    if global_best_routes is None:
        global_best_routes = [[0, 0] for _ in range(truck_count)]
    routes = repair(global_best_routes)
    return routes
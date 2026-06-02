import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    # Constructive heuristic: randomized greedy insertion minimizing max distance increase
    def construct():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_increase = float('inf')
            best_moves = []
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
                    if increase < best_increase - 1e-12:
                        best_increase = increase
                        best_moves = [(r_idx, pos)]
                    elif abs(increase - best_increase) < 1e-12:
                        best_moves.append((r_idx, pos))
            best_route, best_pos = random.choice(best_moves)
            routes[best_route].insert(best_pos, cust)
        return routes
    
    # VND local search: shuffled phases each round
    def local_search(routes, best_routes, best_max):
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
                                    # ensure depot at ends
                                    if new_route1[0] != 0:
                                        new_route1.insert(0, 0)
                                    if new_route1[-1] != 0:
                                        new_route1.append(0)
                                    if new_route2[0] != 0:
                                        new_route2.insert(0, 0)
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
                                    # standard 2-opt*: swap tails
                                    new_route1 = route1[:i] + route2[j:]
                                    new_route2 = route2[:j] + route1[i:]
                                    # ensure depot at start and end
                                    if new_route1[0] != 0:
                                        new_route1.insert(0, 0)
                                    if new_route1[-1] != 0:
                                        new_route1.append(0)
                                    if new_route2[0] != 0:
                                        new_route2.insert(0, 0)
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
        return best_routes, best_max
    
    # Perturbation: remove random block from longest route and reinsert greedily
    def perturb(routes, intensity=1):
        # identify longest route
        max_dist = -1
        max_idx = 0
        for idx, r in enumerate(routes):
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
                max_idx = idx
        route_long = routes[max_idx]
        # remove block of size intensity (min 1, max long route length-2)
        max_block = min(intensity, len(route_long)-2)
        if max_block < 1:
            return routes
        block_len = random.randint(1, max_block)
        start = random.randint(1, len(route_long)-1-block_len)
        block = route_long[start:start+block_len]
        # remove block
        new_route_long = route_long[:start] + route_long[start+block_len:]
        routes[max_idx] = new_route_long
        # reinsert greedily minimizing max distance
        customers = block
        random.shuffle(customers)
        for cust in customers:
            best_increase = float('inf')
            best_moves = []
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
                    if increase < best_increase - 1e-12:
                        best_increase = increase
                        best_moves = [(r_idx, pos)]
                    elif abs(increase - best_increase) < 1e-12:
                        best_moves.append((r_idx, pos))
            best_route, best_pos = random.choice(best_moves)
            routes[best_route].insert(best_pos, cust)
        return routes
    
    global_best_routes = None
    global_best_max = float('inf')
    restarts = max(10, n // 10)
    no_improve_restarts = 0
    for _ in range(restarts):
        routes = construct()
        routes, current_max = local_search(routes, routes, max_distance(routes))
        if current_max < global_best_max - 1e-12:
            global_best_max = current_max
            global_best_routes = [r[:] for r in routes]
            no_improve_restarts = 0
        else:
            no_improve_restarts += 1
        intensity = 1 + (no_improve_restarts // 2)
        if intensity > 5:
            intensity = 5
        # perturb from best solution
        if global_best_routes is None:
            continue
        base = [r[:] for r in global_best_routes]
        routes = perturb(base, intensity)
    
    if global_best_routes is None:
        global_best_routes = [[0, 0] for _ in range(truck_count)]
    # ensure correctness
    for r in global_best_routes:
        if r[0] != 0:
            r.insert(0, 0)
        if r[-1] != 0:
            r.append(0)
    # ensure all customers present
    used = set()
    for r in global_best_routes:
        for c in r:
            if c != 0:
                used.add(c)
    all_customers = set(range(1, n))
    missing = all_customers - used
    if missing:
        # add missing customers to shortest route (will rarely happen)
        min_idx = min(range(truck_count), key=lambda i: len(global_best_routes[i]))
        for c in missing:
            global_best_routes[min_idx].insert(-1, c)
    return global_best_routes
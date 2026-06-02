import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]

    def route_distance(route):
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

    def construct_greedy():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_increase = float('inf')
            best_moves = []
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
                    if increase < best_increase - 1e-12:
                        best_increase = increase
                        best_moves = [(r_idx, pos)]
                    elif abs(increase - best_increase) < 1e-12:
                        best_moves.append((r_idx, pos))
            r_idx, pos = random.choice(best_moves)
            routes[r_idx].insert(pos, cust)
        return routes

    def local_search(routes):
        best_routes = [r[:] for r in routes]
        best_max = max_route_distance(routes)
        improved = True
        max_iter = (n - 1) * truck_count * 2
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # neighborhoods in random order
            neighborhoods = ['intra_2opt', 'relocate', 'swap']
            random.shuffle(neighborhoods)
            for neigh in neighborhoods:
                if neigh == 'intra_2opt':
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
                elif neigh == 'relocate':
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
                elif neigh == 'swap':
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
                if improved:
                    break
        return best_routes

    def perturb(routes, intensity=1):
        for _ in range(intensity):
            candidates = [i for i, r in enumerate(routes) if len(r) >= 3 + intensity]
            if not candidates:
                continue
            src = random.choice(candidates)
            route = routes[src]
            block_len = random.randint(1, min(3, len(route)-2))
            start = random.randint(1, len(route)-1-block_len)
            block = route[start:start+block_len]
            del route[start:start+block_len]
            for cust in block:
                best_increase = float('inf')
                best_moves = []
                current_max = max_route_distance(routes)
                for r_idx in range(truck_count):
                    r = routes[r_idx]
                    for pos in range(1, len(r)):
                        prev = r[pos-1]
                        nxt = r[pos]
                        added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_route_dist = route_distance(r) + added
                        other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                        new_max = max(new_route_dist, other_max)
                        increase = new_max - current_max
                        if increase < best_increase - 1e-12:
                            best_increase = increase
                            best_moves = [(r_idx, pos)]
                        elif abs(increase - best_increase) < 1e-12:
                            best_moves.append((r_idx, pos))
                r_idx, pos = random.choice(best_moves)
                routes[r_idx].insert(pos, cust)
        return routes

    global_best_routes = None
    global_best_max = float('inf')
    restarts = min(10, n)
    no_improve_restarts = 0
    for restart in range(restarts):
        routes = construct_greedy()
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
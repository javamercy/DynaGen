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

    # constructive heuristic: randomized greedy insertion (minimize increase in max route distance)
    def construct():
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
            best_route, best_pos = random.choice(best_moves)
            routes[best_route].insert(best_pos, cust)
        return routes

    # Local search with steepest descent VND
    def local_search(routes):
        best_routes = [r[:] for r in routes]
        best_max = max_route_distance(routes)
        max_passes = 3
        for _ in range(max_passes):
            improved = False
            phases = ['2opt', 'relocate', 'swap', 'cross', '2optstar']
            random.shuffle(phases)
            for phase in phases:
                best_move = None
                best_new_max = best_max
                # Evaluate all moves in neighborhood
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
                                if new_max < best_new_max - 1e-12:
                                    best_new_max = new_max
                                    best_move = ('2opt', r_idx, i, j, new_route)
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
                                    if new_max < best_new_max - 1e-12:
                                        best_new_max = new_max
                                        best_move = ('relocate', src, pos_src, dst, pos_dst, temp_src, new_dst)
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
                                    if new_max < best_new_max - 1e-12:
                                        best_new_max = new_max
                                        best_move = ('swap', t1, i, t2, j, new_route1, new_route2)
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
                                    if new_max < best_new_max - 1e-12:
                                        best_new_max = new_max
                                        best_move = ('cross', t1, i, t2, j, new_route1, new_route2)
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
                                    # Ensure start and end with depot
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
                                    if new_max < best_new_max - 1e-12:
                                        best_new_max = new_max
                                        best_move = ('2optstar', t1, i, t2, j, new_route1, new_route2)
                # Apply best move if found
                if best_move is not None:
                    if best_move[0] == '2opt':
                        _, r_idx, i, j, new_route = best_move
                        routes[r_idx] = new_route
                        best_max = best_new_max
                        improved = True
                        report_best_vrp(routes)
                    elif best_move[0] == 'relocate':
                        _, src, pos_src, dst, pos_dst, temp_src, new_dst = best_move
                        routes[src] = temp_src
                        routes[dst] = new_dst
                        best_max = best_new_max
                        improved = True
                        report_best_vrp(routes)
                    elif best_move[0] == 'swap':
                        _, t1, i, t2, j, new_route1, new_route2 = best_move
                        routes[t1] = new_route1
                        routes[t2] = new_route2
                        best_max = best_new_max
                        improved = True
                        report_best_vrp(routes)
                    elif best_move[0] == 'cross':
                        _, t1, i, t2, j, new_route1, new_route2 = best_move
                        routes[t1] = new_route1
                        routes[t2] = new_route2
                        best_max = best_new_max
                        improved = True
                        report_best_vrp(routes)
                    elif best_move[0] == '2optstar':
                        _, t1, i, t2, j, new_route1, new_route2 = best_move
                        routes[t1] = new_route1
                        routes[t2] = new_route2
                        best_max = best_new_max
                        improved = True
                        report_best_vrp(routes)
            if not improved:
                break
        return routes

    # perturbation: move block of random length (1-5) from one route to another
    def perturb(routes, intensity=1):
        for _ in range(intensity):
            src = random.randint(0, truck_count-1)
            while len(routes[src]) <= 3:
                src = random.randint(0, truck_count-1)
            block_len = random.randint(1, min(5, len(routes[src])-2))
            start = random.randint(1, len(routes[src])-1-block_len)
            block = routes[src][start:start+block_len]
            del routes[src][start:start+block_len]
            dst = random.randint(0, truck_count-1)
            pos = random.randint(1, len(routes[dst])-1)
            for c in block:
                routes[dst].insert(pos, c)
                pos += 1
        return routes

    global_best_routes = None
    global_best_max = float('inf')
    restarts = max(5, n // 10)
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
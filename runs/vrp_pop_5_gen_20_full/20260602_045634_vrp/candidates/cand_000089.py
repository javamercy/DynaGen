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

    # Biased greedy construction: insert customer that minimizes max_route_distance, with random tie-breaking
    def construct():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_max = float('inf')
            best_moves = []
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_route_dist = route_distance(new_route)
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    if new_max < best_max - 1e-12:
                        best_max = new_max
                        best_moves = [(r_idx, pos)]
                    elif abs(new_max - best_max) < 1e-12:
                        best_moves.append((r_idx, pos))
            r_idx, pos = random.choice(best_moves)
            routes[r_idx].insert(pos, cust)
        return routes

    def two_opt(routes, max_route, best_max):
        improved = False
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
                        best_max = new_max
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
            if improved:
                break
        return improved, best_max

    def relocate(routes, best_max):
        improved = False
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
                            best_max = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        return improved, best_max

    def swap(routes, best_max):
        improved = False
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
                            best_max = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        return improved, best_max

    def cross(routes, best_max):
        improved = False
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
                            best_max = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        return improved, best_max

    def two_opt_star(routes, best_max):
        improved = False
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
                        # Standard 2-opt*: swap tails after i and j
                        new_route1 = route1[:i] + route2[j:]
                        new_route2 = route2[:j] + route1[i:]
                        # Ensure depot endpoints
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
                            best_max = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        return improved, best_max

    def vnd(routes):
        best_max = max_route_distance(routes)
        improved = True
        while improved:
            improved = False
            # Random order of neighborhoods for diversity
            phases = ['2opt', 'relocate', 'swap', 'cross', '2optstar']
            random.shuffle(phases)
            for phase in phases:
                if phase == '2opt':
                    imp, best_max = two_opt(routes, best_max, best_max)
                elif phase == 'relocate':
                    imp, best_max = relocate(routes, best_max)
                elif phase == 'swap':
                    imp, best_max = swap(routes, best_max)
                elif phase == 'cross':
                    imp, best_max = cross(routes, best_max)
                elif phase == '2optstar':
                    imp, best_max = two_opt_star(routes, best_max)
                if imp:
                    improved = True
                    break  # restart the loop after any improvement
        return routes

    # Perturbation: transfer a block of consecutive customers from one route to another
    def perturb(routes, intensity):
        new_routes = [r[:] for r in routes]
        for _ in range(intensity):
            # select source route with at least one customer
            src = random.randint(0, truck_count-1)
            while len(new_routes[src]) <= 3:
                src = random.randint(0, truck_count-1)
            route_src = new_routes[src]
            # choose random block length (1 to 3)
            block_len = random.randint(1, min(3, len(route_src)-2))
            start = random.randint(1, len(route_src)-1-block_len)
            block = route_src[start:start+block_len]
            # remove block from source
            del new_routes[src][start:start+block_len]
            # select destination route (can be same? better to be different to avoid trivial)
            dst = random.randint(0, truck_count-1)
            # insert block at random position
            pos = random.randint(1, len(new_routes[dst])-1)
            for c in block:
                new_routes[dst].insert(pos, c)
                pos += 1
        return new_routes

    global_best_routes = None
    global_best_max = float('inf')
    restarts = max(5, n // 10)
    no_improve = 0
    for restart in range(restarts):
        routes = construct()
        routes = vnd(routes)
        current_max = max_route_distance(routes)
        if current_max < global_best_max - 1e-12:
            global_best_max = current_max
            global_best_routes = [r[:] for r in routes]
            no_improve = 0
        else:
            no_improve += 1
        intensity = 1 + (no_improve // 3)
        if restart < restarts - 1:
            # perturb from global best with some probability
            if random.random() < 0.5 and global_best_routes is not None:
                base = [r[:] for r in global_best_routes]
            else:
                base = [r[:] for r in routes]
            routes = perturb(base, intensity)

    if global_best_routes is None:
        global_best_routes = [[0, 0] for _ in range(truck_count)]
    global_best_routes = repair(global_best_routes)
    return global_best_routes
import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customer_count = n - 1

    if truck_count >= customer_count:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        try:
            report_best_vrp(routes)
        except:
            pass
        return routes

    def compute_distance(route):
        if len(route) <= 2:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    def best_insertion(route, cust):
        best_pos = 1
        best_increase = float('inf')
        for pos in range(1, len(route)):
            prev = route[pos-1]
            next_ = route[pos]
            increase = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
            if increase < best_increase - 1e-9:
                best_increase = increase
                best_pos = pos
        return best_pos, best_increase

    def regret_insertion(route, cust, regret_k):
        # compute k-best insertion costs and returns best position and regret
        costs = []
        for pos in range(1, len(route)):
            prev = route[pos-1]
            next_ = route[pos]
            increase = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
            costs.append((increase, pos))
        costs.sort(key=lambda x: x[0])
        if len(costs) < 2:
            return costs[0][1], costs[0][0], 0
        regret = sum(c[0] for c in costs[1:regret_k+1]) - costs[0][0]
        return costs[0][1], costs[0][0], regret

    def generate_initial_solution(seed_offset):
        random.seed(seed_offset)
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        # Seed selection: first seed random, then farthest
        seeds = []
        first_seed = random.choice(range(1, n))
        seeds.append(first_seed)
        unassigned.remove(first_seed)
        while len(seeds) < truck_count and unassigned:
            max_min_dist = -1
            chosen = -1
            for cust in unassigned:
                min_dist = min(distance_matrix[cust][s] for s in seeds)
                if min_dist > max_min_dist + 1e-9:
                    max_min_dist = min_dist
                    chosen = cust
            if chosen != -1:
                seeds.append(chosen)
                unassigned.remove(chosen)
            else:
                break
        while len(seeds) < truck_count and unassigned:
            seeds.append(unassigned.pop(0))
        for t in range(len(seeds)):
            routes[t] = [0, seeds[t], 0]
        # Regret-2 insertion for remaining customers
        while unassigned:
            best_cust = None
            best_route = -1
            best_pos = None
            best_regret = -float('inf')
            for cust in unassigned:
                for t in range(truck_count):
                    if len(routes[t]) == 2:
                        # only one possible insertion
                        regret = 0.0
                        pos = 1
                    else:
                        pos, inc, regret = regret_insertion(routes[t], cust, 2)
                    if regret > best_regret + 1e-9 or (abs(regret - best_regret) < 1e-9 and t < best_route):
                        best_regret = regret
                        best_cust = cust
                        best_route = t
                        best_pos = pos
            # Insert best_cust into best_route at best_pos
            routes[best_route] = routes[best_route][:best_pos] + [best_cust] + routes[best_route][best_pos:]
            unassigned.remove(best_cust)
        return routes

    def local_search_sa(routes, initial_temp, cooling_rate, max_iter):
        distances = [compute_distance(r) for r in routes]
        current_max = max(distances)
        best_routes = [r[:] for r in routes]
        best_max = current_max
        T = initial_temp
        iter_count = 0
        while iter_count < max_iter:
            improved = False
            # Intra-route 2-opt
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                best_route_local = route[:]
                best_dist_local = compute_distance(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_distance(new_route)
                        if new_dist < best_dist_local - 1e-9:
                            best_dist_local = new_dist
                            best_route_local = new_route
                if best_dist_local < compute_distance(routes[t]) - 1e-9:
                    routes[t] = best_route_local
                    improved = True
            if improved:
                distances = [compute_distance(r) for r in routes]
                current_max = max(distances)
                if current_max < best_max - 1e-9:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    try:
                        report_best_vrp(routes)
                    except:
                        pass
                else:
                    # update distances not needed
                    pass
                iter_count += 1
                continue

            # Inter-route relocate
            best_move = None
            best_delta = 0.0
            for src in range(truck_count):
                src_route = routes[src]
                if len(src_route) <= 2:
                    continue
                for cust_idx in range(1, len(src_route)-1):
                    cust = src_route[cust_idx]
                    new_src = src_route[:cust_idx] + src_route[cust_idx+1:]
                    for dst in range(truck_count):
                        if dst == src:
                            continue
                        dst_route = routes[dst]
                        if len(dst_route) == 2:
                            pos = 1
                        else:
                            pos, _ = best_insertion(dst_route, cust)
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        new_dist_src = compute_distance(new_src)
                        new_dist_dst = compute_distance(new_dst)
                        other_dists = [distances[i] for i in range(truck_count) if i not in (src, dst)]
                        new_max = max(new_dist_src, new_dist_dst, *other_dists)
                        delta = new_max - current_max
                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_move = (src, dst, new_src, new_dst)
            if best_move is not None and (best_delta < -1e-9 or random.random() < math.exp(-best_delta / T)):
                src, dst, new_src, new_dst = best_move
                routes[src] = new_src
                routes[dst] = new_dst
                distances = [compute_distance(r) for r in routes]
                current_max = max(distances)
                if current_max < best_max - 1e-9:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    try:
                        report_best_vrp(routes)
                    except:
                        pass
                iter_count += 1
                T *= cooling_rate
                continue

            # Inter-route swap
            best_move = None
            best_delta = 0.0
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
                            new_dist1 = compute_distance(new_route1)
                            new_dist2 = compute_distance(new_route2)
                            other_dists = [distances[k] for k in range(truck_count) if k not in (t1, t2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            delta = new_max - current_max
                            if delta < best_delta - 1e-9:
                                best_delta = delta
                                best_move = (t1, t2, new_route1, new_route2)
            if best_move is not None and (best_delta < -1e-9 or random.random() < math.exp(-best_delta / T)):
                t1, t2, new_route1, new_route2 = best_move
                routes[t1] = new_route1
                routes[t2] = new_route2
                distances = [compute_distance(r) for r in routes]
                current_max = max(distances)
                if current_max < best_max - 1e-9:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    try:
                        report_best_vrp(routes)
                    except:
                        pass
                iter_count += 1
                T *= cooling_rate
                continue

            # Inter-route 2-opt*
            best_move = None
            best_delta = 0.0
            for t1 in range(truck_count):
                for t2 in range(t1+1, truck_count):
                    route1 = routes[t1]
                    route2 = routes[t2]
                    if len(route1) <= 2 or len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-2):
                        for j in range(1, len(route2)-2):
                            new_route1 = route1[:i+1] + route2[j+1:]
                            new_route2 = route2[:j+1] + route1[i+1:]
                            new_dist1 = compute_distance(new_route1)
                            new_dist2 = compute_distance(new_route2)
                            other_dists = [distances[k] for k in range(truck_count) if k not in (t1, t2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            delta = new_max - current_max
                            if delta < best_delta - 1e-9:
                                best_delta = delta
                                best_move = (t1, t2, new_route1, new_route2)
            if best_move is not None and (best_delta < -1e-9 or random.random() < math.exp(-best_delta / T)):
                t1, t2, new_route1, new_route2 = best_move
                routes[t1] = new_route1
                routes[t2] = new_route2
                distances = [compute_distance(r) for r in routes]
                current_max = max(distances)
                if current_max < best_max - 1e-9:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    try:
                        report_best_vrp(routes)
                    except:
                        pass
                iter_count += 1
                T *= cooling_rate
                continue

            # If no improvement, break
            break
        return best_routes

    best_routes = None
    best_max_dist = float('inf')
    num_restarts = 10
    max_iter = n * truck_count * 2
    initial_temp = 10.0
    cooling_rate = 0.995

    for restart in range(num_restarts):
        routes = generate_initial_solution(restart)
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        try:
            report_best_vrp(routes)
        except:
            pass
        routes = local_search_sa(routes, initial_temp, cooling_rate, max_iter)
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        try:
            report_best_vrp(routes)
        except:
            pass
        if max_dist < best_max_dist - 1e-9:
            best_max_dist = max_dist
            best_routes = [route[:] for route in routes]

    if best_routes is None:
        best_routes = [[0,0] for _ in range(truck_count)]
    return best_routes
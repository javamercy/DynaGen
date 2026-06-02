import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    cust_count = n - 1
    
    if truck_count >= cust_count:
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
            nxt = route[pos]
            increase = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
            if increase < best_increase - 1e-9:
                best_increase = increase
                best_pos = pos
        return best_pos, best_increase
    
    def generate_initial_solution(seed):
        random.seed(seed)
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        seeds = []
        first_seed = random.choice(range(1, n))
        seeds.append(first_seed)
        unassigned.remove(first_seed)
        while len(seeds) < truck_count and unassigned:
            max_min_dist = -1
            chosen = -1
            for cust in unassigned:
                min_dist = min(distance_matrix[cust][s] for s in seeds)
                if min_dist > max_min_dist:
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
        random.shuffle(unassigned)
        while unassigned:
            best_cust = None
            best_route = -1
            best_pos = -1
            best_max = float('inf')
            for cust in unassigned:
                for t in range(truck_count):
                    if len(routes[t]) == 2:
                        new_dist = 2 * distance_matrix[0][cust]
                        max_other = max(compute_distance(routes[i]) for i in range(truck_count) if i != t)
                        new_max = max(new_dist, max_other)
                        if new_max < best_max:
                            best_max = new_max
                            best_cust = cust
                            best_route = t
                            best_pos = 1
                    else:
                        for pos in range(1, len(routes[t])):
                            new_route = routes[t][:pos] + [cust] + routes[t][pos:]
                            new_dist = compute_distance(new_route)
                            max_other = max(compute_distance(routes[i]) for i in range(truck_count) if i != t)
                            new_max = max(new_dist, max_other)
                            if new_max < best_max:
                                best_max = new_max
                                best_cust = cust
                                best_route = t
                                best_pos = pos
            if best_cust is not None:
                routes[best_route] = routes[best_route][:best_pos] + [best_cust] + routes[best_route][best_pos:]
                unassigned.remove(best_cust)
            else:
                break
        return routes
    
    def local_search(routes):
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        total_dist = sum(distances)
        improved = True
        max_iter = n * truck_count * 2
        iters = 0
        while improved and iters < max_iter:
            improved = False
            # intra 2-opt
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_route_dist = compute_distance(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_distance(new_route)
                        if new_dist < best_route_dist - 1e-9:
                            best_route_dist = new_dist
                            best_route = new_route
                if best_route_dist < compute_distance(routes[t]) - 1e-9:
                    routes[t] = best_route
                    improved = True
            if improved:
                distances = [compute_distance(r) for r in routes]
                new_max = max(distances)
                new_total = sum(distances)
                if new_max < max_dist - 1e-9 or (abs(new_max - max_dist) < 1e-9 and new_total < total_dist - 1e-9):
                    max_dist = new_max
                    total_dist = new_total
                    try:
                        report_best_vrp(routes)
                    except:
                        pass
                iters += 1
                continue
            # inter relocate
            best_move = None
            best_reduction = 0.0
            current_max = max_dist
            current_total = total_dist
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
                        pos, inc = best_insertion(dst_route, cust)
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        new_dist1 = compute_distance(new_src)
                        new_dist2 = compute_distance(new_dst)
                        others = [distances[i] for i in range(truck_count) if i not in (src, dst)]
                        new_max = max(new_dist1, new_dist2, *others)
                        new_total = new_dist1 + new_dist2 + sum(others)
                        reduction = current_max - new_max
                        if reduction > best_reduction + 1e-9 or (abs(reduction - best_reduction) < 1e-9 and new_total < current_total - 1e-9):
                            best_reduction = reduction
                            best_move = (src, dst, new_src, new_dst, new_max, new_total)
            if best_move is not None and (best_reduction > 1e-9 or best_move[4] < max_dist - 1e-9 or (abs(best_move[4]-max_dist)<1e-9 and best_move[5] < total_dist - 1e-9)):
                src, dst, new_src, new_dst, new_max, new_total = best_move
                routes[src] = new_src
                routes[dst] = new_dst
                distances = [compute_distance(r) for r in routes]
                max_dist = new_max
                total_dist = new_total
                improved = True
                try:
                    report_best_vrp(routes)
                except:
                    pass
                iters += 1
                continue
            # inter swap
            best_move = None
            best_reduction = 0.0
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
                            others = [distances[k] for k in range(truck_count) if k not in (t1, t2)]
                            new_max = max(new_dist1, new_dist2, *others)
                            new_total = new_dist1 + new_dist2 + sum(others)
                            reduction = current_max - new_max
                            if reduction > best_reduction + 1e-9 or (abs(reduction - best_reduction) < 1e-9 and (new_total < current_total - 1e-9 or (abs(new_total - current_total) < 1e-9 and t1*1000+t2 < (best_move[0]*1000+best_move[1] if best_move else 1000000)))):
                                best_reduction = reduction
                                best_move = (t1, t2, new_route1, new_route2, new_max, new_total)
            if best_move is not None and (best_reduction > 1e-9 or best_move[4] < max_dist - 1e-9 or (abs(best_move[4]-max_dist)<1e-9 and best_move[5] < total_dist - 1e-9)):
                t1, t2, new_route1, new_route2, new_max, new_total = best_move
                routes[t1] = new_route1
                routes[t2] = new_route2
                distances = [compute_distance(r) for r in routes]
                max_dist = new_max
                total_dist = new_total
                improved = True
                try:
                    report_best_vrp(routes)
                except:
                    pass
                iters += 1
                continue
            # inter cross (2-opt*)
            best_move = None
            best_reduction = 0.0
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
                            others = [distances[k] for k in range(truck_count) if k not in (t1, t2)]
                            new_max = max(new_dist1, new_dist2, *others)
                            new_total = new_dist1 + new_dist2 + sum(others)
                            reduction = current_max - new_max
                            if reduction > best_reduction + 1e-9 or (abs(reduction - best_reduction) < 1e-9 and (new_total < current_total - 1e-9 or (abs(new_total - current_total) < 1e-9 and t1*1000+t2 < (best_move[0]*1000+best_move[1] if best_move else 1000000)))):
                                best_reduction = reduction
                                best_move = (t1, t2, new_route1, new_route2, new_max, new_total)
            if best_move is not None and (best_reduction > 1e-9 or best_move[4] < max_dist - 1e-9 or (abs(best_move[4]-max_dist)<1e-9 and best_move[5] < total_dist - 1e-9)):
                t1, t2, new_route1, new_route2, new_max, new_total = best_move
                routes[t1] = new_route1
                routes[t2] = new_route2
                distances = [compute_distance(r) for r in routes]
                max_dist = new_max
                total_dist = new_total
                improved = True
                try:
                    report_best_vrp(routes)
                except:
                    pass
                iters += 1
                continue
            break
        # Route balancing: move customers from longest route to others if it reduces max
        improved = True
        while improved:
            improved = False
            dists = [compute_distance(r) for r in routes]
            max_idx = max(range(truck_count), key=lambda t: dists[t])
            min_idx = min(range(truck_count), key=lambda t: dists[t])
            if max_idx == min_idx:
                break
            route_max = routes[max_idx]
            route_min = routes[min_idx]
            best_cust = None
            best_pos = -1
            best_new_dist_max = float('inf')
            best_new_dist_min = float('inf')
            best_new_max = float('inf')
            for i in range(1, len(route_max)-1):
                cust = route_max[i]
                new_route_max = route_max[:i] + route_max[i+1:]
                for pos in range(1, len(route_min)+1):
                    new_route_min = route_min[:pos] + [cust] + route_min[pos:]
                    new_dist_max = compute_distance(new_route_max)
                    new_dist_min = compute_distance(new_route_min)
                    other_dists = [dists[t] for t in range(truck_count) if t not in (max_idx, min_idx)]
                    new_max = max(new_dist_max, new_dist_min, *other_dists)
                    if new_max < best_new_max - 1e-9:
                        best_new_max = new_max
                        best_cust = cust
                        best_pos = pos
                        best_new_dist_max = new_dist_max
                        best_new_dist_min = new_dist_min
            if best_cust is not None and best_new_max < dists[max_idx] - 1e-9:
                # apply move
                routes[max_idx] = [c for c in routes[max_idx] if c != best_cust]
                routes[min_idx] = routes[min_idx][:best_pos] + [best_cust] + routes[min_idx][best_pos:]
                improved = True
                try:
                    report_best_vrp(routes)
                except:
                    pass
        return routes
    
    def ruin_and_recreate(routes, seed):
        random.seed(seed)
        n_cust = n - 1
        if n_cust == 0:
            return routes
        # Choose removal strategy randomly
        strategy = random.choice(['random', 'worst', 'route_concentrated'])
        num_remove = max(1, int(n_cust * (0.2 + 0.2 * random.random())))
        all_custs = []
        for t in range(truck_count):
            for c in routes[t][1:-1]:
                all_custs.append((t, c))
        if len(all_custs) < num_remove:
            num_remove = len(all_custs)
        if strategy == 'random':
            removed = set(random.sample(all_custs, num_remove))
        elif strategy == 'worst':
            # remove customers with highest insertion cost (detour) in their current route
            costs = []
            for (t, c) in all_custs:
                route = routes[t]
                idx = route.index(c)
                prev = route[idx-1]
                nxt = route[idx+1]
                cost = distance_matrix[prev][c] + distance_matrix[c][nxt] - distance_matrix[prev][nxt]
                costs.append((cost, t, c))
            costs.sort(reverse=True, key=lambda x: x[0])
            removed = set()
            for i in range(num_remove):
                _, t, c = costs[i]
                removed.add((t, c))
        else:  # route_concentrated
            # remove contiguous block from the longest route
            dists = [compute_distance(r) for r in routes]
            longest_idx = max(range(truck_count), key=lambda t: dists[t])
            route = routes[longest_idx]
            if len(route) <= 2:
                # fallback to random
                removed = set(random.sample(all_custs, num_remove))
            else:
                block_len = min(num_remove, len(route)-2)
                start = random.randint(1, len(route)-1-block_len)
                removed = set()
                for i in range(start, start+block_len):
                    removed.add((longest_idx, route[i]))
                # if still need more, add random from other routes
                remaining = num_remove - len(removed)
                if remaining > 0:
                    other_custs = [x for x in all_custs if x not in removed]
                    additional = random.sample(other_custs, min(remaining, len(other_custs)))
                    removed.update(additional)
        # Remove customers
        new_routes = [r[:] for r in routes]
        removed_custs = []
        for (t, c) in removed:
            route = new_routes[t]
            idx = route.index(c)
            route.pop(idx)
            removed_custs.append(c)
        # Reinsert using min-max greedy insertion
        random.shuffle(removed_custs)
        while removed_custs:
            best_cust = None
            best_route = -1
            best_pos = -1
            best_new_max = float('inf')
            best_new_total = float('inf')
            for cust in removed_custs:
                for t in range(truck_count):
                    route = new_routes[t]
                    for pos in range(1, len(route)+1):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = compute_distance(new_route)
                        other_dists = [compute_distance(new_routes[i]) for i in range(truck_count) if i != t]
                        new_max = max(new_dist, *other_dists)
                        new_total = new_dist + sum(other_dists)
                        if new_max < best_new_max - 1e-9 or (abs(new_max - best_new_max) < 1e-9 and new_total < best_new_total - 1e-9):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_cust = cust
                            best_route = t
                            best_pos = pos
            if best_cust is not None:
                new_routes[best_route] = new_routes[best_route][:best_pos] + [best_cust] + new_routes[best_route][best_pos:]
                removed_custs.remove(best_cust)
        return new_routes
    
    best_routes = None
    best_max_dist = float('inf')
    best_total_dist = 0.0
    num_restarts = 5
    perturbations_per_restart = 15
    
    for restart in range(num_restarts):
        routes = generate_initial_solution(restart)
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        total_dist = sum(distances)
        try:
            report_best_vrp(routes)
        except:
            pass
        routes = local_search(routes)
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        total_dist = sum(distances)
        try:
            report_best_vrp(routes)
        except:
            pass
        initial_temp = 0.1 * max_dist
        temp = initial_temp
        for pert in range(perturbations_per_restart):
            seed = restart * (perturbations_per_restart + 1) + pert
            perturbed = ruin_and_recreate(routes, seed)
            perturbed = local_search(perturbed)
            dists = [compute_distance(r) for r in perturbed]
            new_max = max(dists)
            new_total = sum(dists)
            delta = new_max - max_dist
            if delta < 0 or (delta == 0 and new_total < total_dist) or random.random() < math.exp(-delta / temp):
                routes = perturbed
                max_dist = new_max
                total_dist = new_total
                if new_max < best_max_dist - 1e-9 or (abs(new_max - best_max_dist) < 1e-9 and new_total < best_total_dist):
                    best_max_dist = new_max
                    best_total_dist = new_total
                    best_routes = [route[:] for route in routes]
                    try:
                        report_best_vrp(routes)
                    except:
                        pass
            temp *= 0.95
        if max_dist < best_max_dist - 1e-9 or (abs(max_dist - best_max_dist) < 1e-9 and total_dist < best_total_dist):
            best_max_dist = max_dist
            best_total_dist = total_dist
            best_routes = [route[:] for route in routes]
    
    if best_routes is None:
        best_routes = generate_initial_solution(0)
        try:
            report_best_vrp(best_routes)
        except:
            pass
    return best_routes
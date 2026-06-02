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
            best_total = float('inf')
            for cust in unassigned:
                for t in range(truck_count):
                    pos, inc = best_insertion(routes[t], cust)
                    new_dist = compute_distance(routes[t][:pos] + [cust] + routes[t][pos:])
                    max_other = max(compute_distance(routes[i]) for i in range(truck_count) if i != t)
                    new_max = max(new_dist, max_other)
                    if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_dist < best_total):
                        best_max = new_max
                        best_total = new_dist
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
                            if reduction > best_reduction + 1e-9 or (abs(reduction - best_reduction) < 1e-9 and new_total < current_total - 1e-9):
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
            # inter cross
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
                            if reduction > best_reduction + 1e-9 or (abs(reduction - best_reduction) < 1e-9 and new_total < current_total - 1e-9):
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
        return routes

    def balance_routes(routes):
        # Explicitly try to reduce the longest route by moving customers to the shortest route
        improved = True
        while improved:
            improved = False
            distances = [compute_distance(r) for r in routes]
            max_idx = max(range(truck_count), key=lambda i: distances[i])
            min_idx = min(range(truck_count), key=lambda i: distances[i])
            if max_idx == min_idx:
                break
            long_route = routes[max_idx]
            short_route = routes[min_idx]
            best_move = None
            best_improvement = 0.0
            for cust_idx in range(1, len(long_route)-1):
                cust = long_route[cust_idx]
                new_long = long_route[:cust_idx] + long_route[cust_idx+1:]
                pos, inc = best_insertion(short_route, cust)
                new_short = short_route[:pos] + [cust] + short_route[pos:]
                new_long_dist = compute_distance(new_long)
                new_short_dist = compute_distance(new_short)
                new_max = max(new_long_dist, new_short_dist, *[distances[i] for i in range(truck_count) if i not in (max_idx, min_idx)])
                improvement = distances[max_idx] - new_max
                if improvement > best_improvement + 1e-9:
                    best_improvement = improvement
                    best_move = (max_idx, min_idx, new_long, new_short)
            if best_move is not None and best_improvement > 1e-9:
                src, dst, new_long, new_short = best_move
                routes[src] = new_long
                routes[dst] = new_short
                improved = True
                try:
                    report_best_vrp(routes)
                except:
                    pass
        return routes

    def ruin_and_recreate(routes, seed, ruin_type, current_max, best_global_max):
        random.seed(seed)
        n_cust = n - 1
        if n_cust == 0:
            return routes
        all_custs = []
        for t in range(truck_count):
            for c in routes[t][1:-1]:
                all_custs.append((t, c))
        if not all_custs:
            return routes
        # Increased removal percentage: 25% to 50%
        num_remove = max(1, int(n_cust * (0.25 + 0.25 * random.random())))
        if len(all_custs) < num_remove:
            num_remove = len(all_custs)
        if ruin_type == 0:  # random ruin
            removed = set(random.sample(all_custs, num_remove))
        elif ruin_type == 1:  # worst route distance contribution
            cust_scores = []
            for t, c in all_custs:
                route = routes[t]
                idx = route.index(c)
                prev = route[idx-1]
                nxt = route[idx+1]
                contrib = distance_matrix[prev][c] + distance_matrix[c][nxt] - (distance_matrix[prev][nxt] if len(route)>2 else 0)
                cust_scores.append(((t,c), contrib))
            cust_scores.sort(key=lambda x: -x[1])
            removed = set(p[0] for p in cust_scores[:num_remove])
        elif ruin_type == 2:  # remove from longest route
            dists = [compute_distance(r) for r in routes]
            max_idx = max(range(truck_count), key=lambda i: dists[i])
            route = routes[max_idx]
            if len(route) > 3:
                start = random.randint(1, len(route)-2)
                length = min(num_remove, len(route)-1-start)
                removed = {(max_idx, route[i]) for i in range(start, start+length)}
                if len(removed) < num_remove:
                    remaining = [p for p in all_custs if p not in removed]
                    extra = random.sample(remaining, min(num_remove-len(removed), len(remaining)))
                    removed.update(extra)
            else:
                removed = set(random.sample(all_custs, num_remove))
        elif ruin_type == 3:  # random sequence removal from random route
            dists = [compute_distance(r) for r in routes]
            tgt = random.randint(0, truck_count-1)
            route = routes[tgt]
            if len(route) > 3:
                start = random.randint(1, len(route)-2)
                length = min(num_remove, len(route)-1-start)
                removed = {(tgt, route[i]) for i in range(start, start+length)}
                if len(removed) < num_remove:
                    remaining = [p for p in all_custs if p not in removed]
                    extra = random.sample(remaining, min(num_remove-len(removed), len(remaining)))
                    removed.update(extra)
            else:
                removed = set(random.sample(all_custs, num_remove))
        else:
            removed = set(random.sample(all_custs, num_remove))
        # remove customers
        new_routes = [r[:] for r in routes]
        removed_custs = []
        for (t, c) in removed:
            route = new_routes[t]
            idx = route.index(c)
            route.pop(idx)
            removed_custs.append(c)
        # reinsert using regret-3 with max-distance cost
        random.shuffle(removed_custs)
        while removed_custs:
            best_cust = None
            best_route = -1
            best_pos = -1
            best_regret = -1e9
            best_cost = None
            for cust in removed_custs:
                costs = []
                for t in range(truck_count):
                    pos, inc = best_insertion(new_routes[t], cust)
                    new_route_dist = compute_distance(new_routes[t][:pos] + [cust] + new_routes[t][pos:])
                    max_other = max(compute_distance(new_routes[i]) for i in range(truck_count) if i != t) if truck_count > 1 else 0
                    new_max = max(new_route_dist, max_other)
                    costs.append((new_max, inc, t, pos))
                costs.sort(key=lambda x: (x[0], x[1]))
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = 0
                if regret > best_regret + 1e-9:
                    best_regret = regret
                    best_cust = cust
                    best_cost = costs[0][0]
                    best_route = costs[0][2]
                    best_pos = costs[0][3]
                elif abs(regret - best_regret) < 1e-9 and costs[0][1] < best_cost:
                    best_cust = cust
                    best_cost = costs[0][1]
                    best_route = costs[0][2]
                    best_pos = costs[0][3]
            if best_cust is not None:
                new_routes[best_route] = new_routes[best_route][:best_pos] + [best_cust] + new_routes[best_route][best_pos:]
                removed_custs.remove(best_cust)
        return new_routes

    best_routes = None
    best_max_dist = float('inf')
    best_total_dist = 0.0
    num_restarts = 3
    perturbations_per_restart = 25  # increased from 20

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
        routes = balance_routes(routes)  # add balancing
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        total_dist = sum(distances)
        try:
            report_best_vrp(routes)
        except:
            pass
        best_global_max = best_max_dist
        threshold = 0.15  # initial record-to-record threshold (increased for more exploration)
        # bias towards longest-route removal (ruin type 2)
        ruin_scores = [1.0, 1.0, 2.0, 1.0]  # higher weight for longest-route ruin
        for pert in range(perturbations_per_restart):
            seed = restart * (perturbations_per_restart + 1) + pert
            total_score = sum(ruin_scores)
            prob = [s/total_score for s in ruin_scores]
            r = random.random()
            cum = 0.0
            chosen = 0
            for i, p in enumerate(prob):
                cum += p
                if r < cum:
                    chosen = i
                    break
            perturbed = ruin_and_recreate(routes, seed, chosen, max_dist, best_global_max)
            perturbed = local_search(perturbed)
            perturbed = balance_routes(perturbed)  # add balancing after LS
            dists = [compute_distance(r) for r in perturbed]
            new_max = max(dists)
            new_total = sum(dists)
            # Record-to-record acceptance: accept if new_max <= best_global_max * (1 + threshold)
            accept = False
            if new_max < max_dist - 1e-9 or (abs(new_max - max_dist) < 1e-9 and new_total < total_dist):
                accept = True
            elif new_max <= best_global_max * (1 + threshold):
                accept = True
            if accept:
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
                if new_max < max_dist - 1e-9:
                    ruin_scores[chosen] *= 1.2
                else:
                    ruin_scores[chosen] *= 0.9
            else:
                ruin_scores[chosen] *= 0.95
            # Decay threshold slower
            threshold *= 0.99
        if max_dist < best_max_dist - 1e-9 or (abs(max_dist - best_max_dist) < 1e-9 and total_dist < best_total_dist):
            best_max_dist = max_dist
            best_total_dist = total_dist
            best_routes = [route[:] for route in routes]

    if best_routes is None:
        best_routes = generate_initial_solution(0)

    return best_routes
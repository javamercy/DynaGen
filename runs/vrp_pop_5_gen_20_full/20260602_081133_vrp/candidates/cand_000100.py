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
        # Assign first truck_count seeds by farthest from depot
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
        # Regret-3 insertion with balance bias
        random.shuffle(unassigned)
        while unassigned:
            best_cust = None
            best_route = -1
            best_pos = -1
            best_value = float('inf')
            best_regret = -1e9
            for cust in unassigned:
                insertion_list = []
                for t in range(truck_count):
                    pos, inc = best_insertion(routes[t], cust)
                    new_route = routes[t][:pos] + [cust] + routes[t][pos:]
                    new_dist = compute_distance(new_route)
                    max_other = max(compute_distance(routes[i]) for i in range(truck_count) if i != t) or 0
                    new_max = max(new_dist, max_other)
                    # secondary metric: number of customers in route
                    load = len(routes[t]) - 2
                    insertion_list.append((new_max, inc, t, pos, load))
                insertion_list.sort(key=lambda x: (x[0], x[4], x[1]))  # sort by new_max, then load, then inc
                if len(insertion_list) >= 3:
                    regret = insertion_list[2][0] - insertion_list[0][0]
                else:
                    regret = insertion_list[-1][0] - insertion_list[0][0]
                best_for_cust = insertion_list[0]
                # Choose customer with max regret, tie on best value
                if regret > best_regret + 1e-9:
                    best_regret = regret
                    best_cust = cust
                    best_route = best_for_cust[2]
                    best_pos = best_for_cust[3]
                    best_value = best_for_cust[0]
                elif abs(regret - best_regret) < 1e-9 and best_for_cust[1] < best_value:
                    best_cust = cust
                    best_route = best_for_cust[2]
                    best_pos = best_for_cust[3]
                    best_value = best_for_cust[1]
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
            # Intra-route 2-opt
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
            # Inter-route relocate
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
            # Inter-route swap
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
            # Inter-route cross
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
        max_iter = n * truck_count
        for _ in range(max_iter):
            distances = [compute_distance(r) for r in routes]
            max_dist = max(distances)
            longest_idx = distances.index(max_dist)
            longest_route = routes[longest_idx]
            if len(longest_route) <= 2:
                break
            best_relocation = None
            best_reduction = 0.
            for cust_idx in range(1, len(longest_route)-1):
                cust = longest_route[cust_idx]
                new_longest = longest_route[:cust_idx] + longest_route[cust_idx+1:]
                new_longest_dist = compute_distance(new_longest)
                for other_idx in range(truck_count):
                    if other_idx == longest_idx:
                        continue
                    other_route = routes[other_idx]
                    pos, inc = best_insertion(other_route, cust)
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_other_dist = compute_distance(new_other)
                    new_max = max(new_longest_dist, new_other_dist, *[distances[i] for i in range(truck_count) if i not in (longest_idx, other_idx)])
                    reduction = max_dist - new_max
                    if reduction > best_reduction + 1e-9:
                        best_reduction = reduction
                        best_relocation = (longest_idx, other_idx, new_longest, new_other, new_max)
            if best_relocation and best_reduction > 1e-9:
                src, dst, new_src, new_dst, new_max = best_relocation
                routes[src] = new_src
                routes[dst] = new_dst
                try:
                    report_best_vrp(routes)
                except:
                    pass
                continue
            # Try swap moves
            best_swap = None
            best_reduction = 0.
            for other_idx in range(truck_count):
                if other_idx == longest_idx:
                    continue
                other_route = routes[other_idx]
                for i in range(1, len(longest_route)-1):
                    for j in range(1, len(other_route)-1):
                        cust1 = longest_route[i]
                        cust2 = other_route[j]
                        new_longest = longest_route[:i] + [cust2] + longest_route[i+1:]
                        new_other = other_route[:j] + [cust1] + other_route[j+1:]
                        new_dist1 = compute_distance(new_longest)
                        new_dist2 = compute_distance(new_other)
                        new_max = max(new_dist1, new_dist2, *[distances[k] for k in range(truck_count) if k not in (longest_idx, other_idx)])
                        reduction = max_dist - new_max
                        if reduction > best_reduction + 1e-9:
                            best_reduction = reduction
                            best_swap = (longest_idx, other_idx, new_longest, new_other, new_max)
            if best_swap and best_reduction > 1e-9:
                src, dst, new_src, new_dst, new_max = best_swap
                routes[src] = new_src
                routes[dst] = new_dst
                try:
                    report_best_vrp(routes)
                except:
                    pass
                continue
            break
        return routes

    def ruin_and_recreate(routes, seed, ruin_type):
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
        num_remove = max(1, int(n_cust * (0.2 + 0.2 * random.random())))
        if len(all_custs) < num_remove:
            num_remove = len(all_custs)
        if ruin_type == 0:  # random
            removed = set(random.sample(all_custs, num_remove))
        elif ruin_type == 1:  # worst
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
        elif ruin_type == 2:  # cluster on longest route
            # Find the route with max distance
            dists = [compute_distance(r) for r in routes]
            max_dist = max(dists)
            target_route_idx = dists.index(max_dist)
            target_route = routes[target_route_idx]
            if len(target_route) > 3:
                start = random.randint(1, len(target_route)-2)
                length = min(num_remove, len(target_route)-1-start)
                removed = {(target_route_idx, target_route[i]) for i in range(start, start+length)}
                if len(removed) < num_remove:
                    remaining = [p for p in all_custs if p not in removed]
                    extra = random.sample(remaining, min(num_remove-len(removed), len(remaining)))
                    removed.update(extra)
            else:
                removed = set(random.sample(all_custs, num_remove))
        else:
            removed = set(random.sample(all_custs, num_remove))
        new_routes = [r[:] for r in routes]
        removed_custs = []
        for (t, c) in removed:
            route = new_routes[t]
            idx = route.index(c)
            route.pop(idx)
            removed_custs.append(c)
        random.shuffle(removed_custs)
        while removed_custs:
            best_cust = None
            best_route = -1
            best_pos = -1
            best_regret = -1e9
            best_cost = None
            for cust in removed_custs:
                insertion_info = []
                for t in range(truck_count):
                    pos, inc = best_insertion(new_routes[t], cust)
                    new_route_dist = compute_distance(new_routes[t][:pos] + [cust] + new_routes[t][pos:])
                    max_other = max(compute_distance(new_routes[i]) for i in range(truck_count) if i != t) or 0
                    new_max = max(new_route_dist, max_other)
                    load = len(new_routes[t]) - 2
                    insertion_info.append((new_max, inc, t, pos, load))
                insertion_info.sort(key=lambda x: (x[0], x[4], x[1]))
                if len(insertion_info) >= 2:
                    regret = insertion_info[1][0] - insertion_info[0][0]
                else:
                    regret = 0
                if regret > best_regret + 1e-9:
                    best_regret = regret
                    best_cust = cust
                    best_route = insertion_info[0][2]
                    best_pos = insertion_info[0][3]
                    best_cost = insertion_info[0][1]
                elif abs(regret - best_regret) < 1e-9 and insertion_info[0][1] < best_cost:
                    best_cust = cust
                    best_route = insertion_info[0][2]
                    best_pos = insertion_info[0][3]
                    best_cost = insertion_info[0][1]
            if best_cust is not None:
                new_routes[best_route] = new_routes[best_route][:best_pos] + [best_cust] + new_routes[best_route][best_pos:]
                removed_custs.remove(best_cust)
        return new_routes

    def shake_swap(routes, seed):
        random.seed(seed)
        routes_with_cust = [t for t in range(truck_count) if len(routes[t]) > 2]
        if len(routes_with_cust) < 2:
            return routes
        t1, t2 = random.sample(routes_with_cust, 2)
        route1 = routes[t1]
        route2 = routes[t2]
        idx1 = random.randint(1, len(route1)-2)
        idx2 = random.randint(1, len(route2)-2)
        new_route1 = route1[:idx1] + [route2[idx2]] + route1[idx1+1:]
        new_route2 = route2[:idx2] + [route1[idx1]] + route2[idx2+1:]
        new_routes = [r[:] for r in routes]
        new_routes[t1] = new_route1
        new_routes[t2] = new_route2
        return new_routes

    best_routes = None
    best_max_dist = float('inf')
    best_total_dist = 0.0
    num_restarts = min(5, max(3, n // 20))
    perturbations_per_restart = min(30, max(10, n))

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
        routes = balance_routes(routes)
        distances = [compute_distance(r) for r in routes]
        max_dist = max(distances)
        total_dist = sum(distances)
        try:
            report_best_vrp(routes)
        except:
            pass
        initial_temp = 0.1 * max_dist
        temp = initial_temp
        ruin_scores = [1.0, 1.0, 1.0]
        no_improve = 0
        for pert in range(perturbations_per_restart):
            seed = restart * (perturbations_per_restart + 1) + pert
            if pert % 5 == 4 and no_improve > 2:
                perturbed = shake_swap(routes, seed)
            else:
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
                perturbed = ruin_and_recreate(routes, seed, chosen)
            perturbed = local_search(perturbed)
            perturbed = balance_routes(perturbed)
            dists = [compute_distance(r) for r in perturbed]
            new_max = max(dists)
            new_total = sum(dists)
            delta = new_max - max_dist
            accept = False
            if delta < 0 or (delta == 0 and new_total < total_dist):
                accept = True
                no_improve = 0
            elif random.random() < math.exp(-delta / temp):
                accept = True
                no_improve += 1
            else:
                no_improve += 1
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
                improvement = delta
                if improvement > 0:
                    ruin_scores[chosen] *= 1.2
                else:
                    ruin_scores[chosen] *= 0.9
            else:
                ruin_scores[chosen] *= 0.95
            if no_improve > 10:
                temp = initial_temp
                no_improve = 0
            else:
                temp *= 0.95
        if max_dist < best_max_dist - 1e-9 or (abs(max_dist - best_max_dist) < 1e-9 and total_dist < best_total_dist):
            best_max_dist = max_dist
            best_total_dist = total_dist
            best_routes = [route[:] for route in routes]

    if best_routes is None:
        best_routes = generate_initial_solution(0)

    return best_routes
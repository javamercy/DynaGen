import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    max_dist = np.max(distance_matrix)

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(route, max_iter=5):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
        return route

    def balance_routes(routes, lengths):
        improved = True
        max_balance_iter = n
        it = 0
        while improved and it < max_balance_iter:
            improved = False
            it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_insert_pos = None
            best_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_insertion_len = float('inf')
                best_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_insertion_len:
                        best_insertion_len = l
                        best_pos = p
                new_min_route = min_route[:best_pos] + [cust] + min_route[best_pos:]
                new_min_len = route_distance(new_min_route)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cust = cust
                    best_insert_pos = best_pos
            if best_cust is not None and best_reduction > 0:
                cust = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:best_insert_pos] + [cust] + min_route[best_insert_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_distance(new_max)
                lengths[min_idx] = route_distance(new_min)
                improved = True
        return routes, lengths

    def regret_insertion_construction(k=3):
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        noise = random.uniform(0, 0.05 * max_dist)
                        incs.append((inc + noise, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= k:
                    regret = sum(incs[i][0] - incs[0][0] for i in range(1, k))
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        lengths = [route_distance(r) for r in routes]
        return routes, lengths

    def split_permutation(perm):
        # not used in ALNS; here for completeness
        pass

    def vnd(routes, lengths):
        improved = True
        max_cycles = 10
        cycle = 0
        while improved and cycle < max_cycles:
            improved = False
            cycle += 1
            # Inter-route relocate
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            for cust in range(1, n):
                src_idx = None
                src_pos = None
                for r_idx, route in enumerate(routes):
                    if cust in route:
                        src_idx = r_idx
                        src_pos = route.index(cust)
                        break
                if src_idx is None:
                    continue
                new_src_route = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
                src_len = route_distance(new_src_route)
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    if len(dst_route) <= 2:
                        continue
                    for ins_pos in range(1, len(dst_route)):
                        new_dst_route = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                        new_lengths = lengths[:]
                        new_lengths[src_idx] = src_len
                        new_lengths[dst_idx] = route_distance(new_dst_route)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_total < best_total) or
                            (new_max == best_new_max and new_total == best_total and src_idx < dst_idx)):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('relocate', src_idx, src_pos, dst_idx, ins_pos, new_src_route, new_dst_route)
            if best_move is not None and best_new_max < max(lengths):
                routes[best_move[1]] = best_move[5]
                routes[best_move[3]] = best_move[6]
                lengths = [route_distance(r) for r in routes]
                improved = True
                continue
            # Inter-route swap
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            for i_idx in range(truck_count):
                i_route = routes[i_idx]
                if len(i_route) <= 2:
                    continue
                for i_pos in range(1, len(i_route)-1):
                    cust_i = i_route[i_pos]
                    for j_idx in range(i_idx+1, truck_count):
                        j_route = routes[j_idx]
                        if len(j_route) <= 2:
                            continue
                        for j_pos in range(1, len(j_route)-1):
                            cust_j = j_route[j_pos]
                            new_i_route = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                            new_j_route = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                            new_lengths = lengths[:]
                            new_lengths[i_idx] = route_distance(new_i_route)
                            new_lengths[j_idx] = route_distance(new_j_route)
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if (new_max < best_new_max or
                                (new_max == best_new_max and new_total < best_total) or
                                (new_max == best_new_max and new_total == best_total and i_idx < j_idx)):
                                best_new_max = new_max
                                best_total = new_total
                                best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i_route, new_j_route)
            if best_move is not None and best_new_max < max(lengths):
                routes[best_move[1]] = best_move[5]
                routes[best_move[3]] = best_move[6]
                lengths = [route_distance(r) for r in routes]
                improved = True
                continue
            # Intra-route 2-opt
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = route_distance(new_route)
                        if new_len >= lengths[r_idx]:
                            continue
                        new_lengths = lengths[:]
                        new_lengths[r_idx] = new_len
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_total < best_total) or
                            (new_max == best_new_max and new_total == best_total and r_idx < 0)):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('2opt', r_idx, i, j, new_route)
            if best_move is not None and best_new_max < max(lengths):
                routes[best_move[1]] = best_move[4]
                lengths = [route_distance(r) for r in routes]
                improved = True
        return routes, lengths

    def evaluate(routes):
        lengths = [route_distance(r) for r in routes]
        return max(lengths), sum(lengths)

    # ALNS operators
    def destroy_random(routes, num_remove):
        # remove random customers
        all_cust = [c for route in routes for c in route[1:-1]]
        if len(all_cust) < num_remove:
            num_remove = len(all_cust)
        random.shuffle(all_cust)
        removed = all_cust[:num_remove]
        new_routes = [[0,0] for _ in range(truck_count)]
        for r_idx, route in enumerate(routes):
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            new_routes[r_idx] = new_route
        return new_routes, removed

    def destroy_worst(routes, num_remove):
        # remove customers with highest saving (contribution)
        savings = []
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                nxt = route[pos+1]
                saving = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                savings.append((saving, cust, r_idx, pos))
        savings.sort(reverse=True)
        removed = [item[1] for item in savings[:num_remove]]
        new_routes = [[0,0] for _ in range(truck_count)]
        for r_idx, route in enumerate(routes):
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            new_routes[r_idx] = new_route
        return new_routes, removed

    def destroy_route(routes, num_remove):
        # remove entire routes that have the highest max?
        # we'll remove customers from the route with highest max length
        lengths = [route_distance(r) for r in routes]
        max_route_idx = max(range(truck_count), key=lambda i: lengths[i])
        route = routes[max_route_idx]
        if len(route) <= 2:
            return destroy_random(routes, num_remove)
        removed = []
        # remove all customers from that route? but we only need num_remove
        # take from worst route until we have enough
        custs = route[1:-1]
        random.shuffle(custs)
        removed = custs[:min(num_remove, len(custs))]
        new_routes = [[0,0] for _ in range(truck_count)]
        for r_idx, route in enumerate(routes):
            if r_idx == max_route_idx:
                new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            else:
                new_route = route[:]
            new_routes[r_idx] = new_route
        # if we didn't remove enough, remove more from other routes
        if len(removed) < num_remove:
            additional = num_remove - len(removed)
            all_remaining = [c for route in new_routes for c in route[1:-1]]
            if additional > len(all_remaining):
                additional = len(all_remaining)
            random.shuffle(all_remaining)
            extra_removed = all_remaining[:additional]
            removed.extend(extra_removed)
            for r_idx, route in enumerate(new_routes):
                new_route = [0] + [c for c in route[1:-1] if c not in extra_removed] + [0]
                new_routes[r_idx] = new_route
        return new_routes, removed

    def repair_greedy(routes, unvisited):
        # greedily insert each customer in the best position
        while unvisited:
            best_cost = float('inf')
            best_cust = None
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        noise = random.uniform(0, 0.05 * max_dist)
                        total = inc + noise
                        if total < best_cost:
                            best_cost = total
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
            if best_cust is not None:
                routes[best_route_idx].insert(best_pos, best_cust)
                unvisited.remove(best_cust)
        return routes

    def repair_regret2(routes, unvisited):
        # regret-2 insertion
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        noise = random.uniform(0, 0.05 * max_dist)
                        incs.append((inc + noise, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= 2:
                    regret = incs[1][0] - incs[0][0]
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            if best_cust is not None:
                routes[best_route_idx].insert(best_pos, best_cust)
                unvisited.remove(best_cust)
        return routes

    # Initial solution
    routes, lengths = regret_insertion_construction(3)
    best_routes = [r[:] for r in routes]
    best_max, best_total = evaluate(routes)
    report_best_vrp(best_routes)

    # ALNS parameters
    max_iter = min(5000, 200 * n)
    temp = 100.0 * max_dist
    cooling_rate = 0.99
    min_temp = 1.0
    destroy_ops = [destroy_random, destroy_worst, destroy_route]
    repair_ops = [repair_greedy, repair_regret2]
    weights = [[1.0]*len(destroy_ops), [1.0]*len(repair_ops)]
    scores = [ [0]*len(destroy_ops), [0]*len(repair_ops) ]
    counts = [ [0]*len(destroy_ops), [0]*len(repair_ops) ]
    current_routes = [r[:] for r in routes]
    current_max, current_total = best_max, best_total
    stagnation = 0
    for iteration in range(max_iter):
        # select destroy and repair adaptively
        # roulette wheel based on weights
        dest_idx = random.choices(range(len(destroy_ops)), weights=weights[0])[0]
        rep_idx = random.choices(range(len(repair_ops)), weights=weights[1])[0]
        # determine removal size
        num_remove = max(1, int( (n-1) * random.uniform(0.1, 0.4) ))
        # destroy
        new_routes, removed = destroy_ops[dest_idx](current_routes, num_remove)
        # repair
        unvisited = set(removed)
        new_routes = repair_ops[rep_idx](new_routes, unvisited)
        # apply VND and balance
        lengths = [route_distance(r) for r in new_routes]
        new_routes, lengths = vnd(new_routes, lengths)
        new_routes, lengths = balance_routes(new_routes, lengths)
        new_max, new_total = evaluate(new_routes)
        # acceptance: simulated annealing
        if new_max < current_max or (new_max == current_max and new_total < current_total):
            accept = True
        else:
            delta = (new_max - current_max) * max_dist + (new_total - current_total)
            if delta <= 0:
                accept = True
            else:
                prob = np.exp(-delta / temp)
                accept = random.random() < prob
        if accept:
            current_routes = [r[:] for r in new_routes]
            current_max, current_total = new_max, new_total
            # update scores
            scores[0][dest_idx] += 1
            scores[1][rep_idx] += 1
            if new_max < best_max or (new_max == best_max and new_total < best_total):
                best_routes = [r[:] for r in new_routes]
                best_max, best_total = new_max, new_total
                report_best_vrp(best_routes)
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
        # update weights every 100 iterations
        if (iteration+1) % 100 == 0:
            for i in range(len(destroy_ops)):
                if counts[0][i] > 0:
                    weights[0][i] = weights[0][i] * 0.9 + 0.1 * (scores[0][i] / counts[0][i])
                else:
                    weights[0][i] = weights[0][i] * 0.9
                scores[0][i] = 0
                counts[0][i] = 0
            for i in range(len(repair_ops)):
                if counts[1][i] > 0:
                    weights[1][i] = weights[1][i] * 0.9 + 0.1 * (scores[1][i] / counts[1][i])
                else:
                    weights[1][i] = weights[1][i] * 0.9
                scores[1][i] = 0
                counts[1][i] = 0
            # reset counts not necessary
        # cooling
        temp = max(temp * cooling_rate, min_temp)
        # restart if stagnation > 200
        if stagnation > 200:
            # reinitialize from scratch
            new_routes, _ = regret_insertion_construction(3)
            lengths = [route_distance(r) for r in new_routes]
            new_routes, lengths = vnd(new_routes, lengths)
            new_routes, lengths = balance_routes(new_routes, lengths)
            new_max, new_total = evaluate(new_routes)
            current_routes = [r[:] for r in new_routes]
            current_max, current_total = new_max, new_total
            if current_max < best_max or (current_max == best_max and current_total < best_total):
                best_routes = [r[:] for r in current_routes]
                best_max, best_total = current_max, current_total
                report_best_vrp(best_routes)
            stagnation = 0
        # also update counts for selected operators
        counts[0][dest_idx] += 1
        counts[1][rep_idx] += 1

    return best_routes
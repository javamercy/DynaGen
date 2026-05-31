import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
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

    def split_permutation(perm):
        m = len(perm)
        if m == 0:
            return [[0,0] for _ in range(truck_count)], [0.0]*truck_count
        # Precompute prefix distances: dist[i][j] = distance of segment perm[i:j] (i<j)
        # Also from depot to first and last to depot
        seg = [[0.0]* (m+1) for _ in range(m)]
        for i in range(m):
            for j in range(i+1, m+1):
                if j == i+1:
                    seg[i][j] = distance_matrix[depot, perm[i]] + distance_matrix[perm[i], depot]
                else:
                    seg[i][j] = seg[i][j-1] - distance_matrix[perm[j-2], depot] + distance_matrix[perm[j-2], perm[j-1]] + distance_matrix[perm[j-1], depot]
        # DP: dp[i][k] = min max distance for first i customers with k trucks
        dp = [[float('inf')]*(truck_count+1) for _ in range(m+1)]
        dp[0][0] = 0.0
        for i in range(1, m+1):
            for k in range(1, min(i, truck_count)+1):
                best = float('inf')
                for j in range(k-1, i):
                    cost = seg[j][i]
                    val = max(dp[j][k-1], cost)
                    if val < best:
                        best = val
                dp[i][k] = best
        # Backtrack to get routes
        routes = [[0,0] for _ in range(truck_count)]
        i = m
        k = truck_count
        while i > 0 and k > 0:
            best_j = -1
            best_val = float('inf')
            for j in range(k-1, i):
                val = max(dp[j][k-1], seg[j][i])
                if val < best_val - 1e-9:
                    best_val = val
                    best_j = j
            # route from best_j to i-1
            route = [depot] + perm[best_j:i] + [depot]
            routes[k-1] = route
            i = best_j
            k -= 1
        lengths = [route_distance(r) for r in routes]
        return routes, lengths

    def local_search(routes, lengths):
        improved = True
        max_cycles = 10
        cycle = 0
        while improved and cycle < max_cycles:
            improved = False
            cycle += 1
            # relocate
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            for cust in range(1, n):
                src_idx = None
                src_pos = None
                for i, route in enumerate(routes):
                    if cust in route:
                        src_idx = i
                        src_pos = route.index(cust)
                        break
                if src_idx is None:
                    continue
                new_src = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
                src_len = route_distance(new_src)
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    if len(routes[dst_idx]) <= 2:
                        continue
                    for ins_pos in range(1, len(routes[dst_idx])):
                        new_dst = routes[dst_idx][:ins_pos] + [cust] + routes[dst_idx][ins_pos:]
                        new_lengths = lengths[:]
                        new_lengths[src_idx] = src_len
                        new_lengths[dst_idx] = route_distance(new_dst)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('relocate', src_idx, src_pos, dst_idx, ins_pos, new_src, new_dst)
            if best_move is not None and best_new_max < max(lengths):
                routes[best_move[1]] = best_move[5]
                routes[best_move[3]] = best_move[6]
                lengths = [route_distance(r) for r in routes]
                improved = True
                continue
            # swap
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
                            new_i = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                            new_j = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                            new_lengths = lengths[:]
                            new_lengths[i_idx] = route_distance(new_i)
                            new_lengths[j_idx] = route_distance(new_j)
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                                best_new_max = new_max
                                best_total = new_total
                                best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i, new_j)
            if best_move is not None and best_new_max < max(lengths):
                routes[best_move[1]] = best_move[5]
                routes[best_move[3]] = best_move[6]
                lengths = [route_distance(r) for r in routes]
                improved = True
                continue
            # 2-opt
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
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('2opt', r_idx, i, j, new_route)
            if best_move is not None and best_new_max < max(lengths):
                routes[best_move[1]] = best_move[4]
                lengths = [route_distance(r) for r in routes]
                improved = True
        return routes, lengths

    # Initialize pheromone trails (n x n matrix including depot)
    tau = np.ones((n, n)) * 0.01
    eta = 1.0 / (distance_matrix + 1e-10)
    alpha = 1.0
    beta = 2.0
    evaporation = 0.1
    num_ants = max(5, min(20, n))
    iterations = max(10, min(30, n*2))

    # Best solution
    best_routes = None
    best_max = float('inf')
    best_total = float('inf')

    # Initial best from regret-3 construction (from parent genes)
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
                        noise = random.uniform(0, 0.1 * max_dist)
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
        routes, lengths = balance_routes(routes, lengths)
        return routes, lengths

    init_routes, init_lengths = regret_insertion_construction(3)
    init_max = max(init_lengths)
    init_total = sum(init_lengths)
    if init_max < best_max or (init_max == best_max and init_total < best_total):
        best_max = init_max
        best_total = init_total
        best_routes = [r[:] for r in init_routes]
        report_best_vrp(best_routes)

    for it in range(iterations):
        ant_solutions = []
        for ant in range(num_ants):
            # Build permutation using pheromone and heuristic
            unvisited = set(customers)
            current = depot
            perm = []
            while unvisited:
                # Compute probabilities for next customer
                candidates = list(unvisited)
                probs = []
                denom = 0.0
                for j in candidates:
                    p = (tau[current, j] ** alpha) * (eta[current, j] ** beta)
                    probs.append(p)
                    denom += p
                if denom == 0:
                    # fallback to random
                    j = random.choice(candidates)
                else:
                    probs = [p/denom for p in probs]
                    j = random.choices(candidates, weights=probs, k=1)[0]
                perm.append(j)
                unvisited.remove(j)
                current = j
            # Split permutation into routes
            routes, lengths = split_permutation(perm)
            routes, lengths = local_search(routes, lengths)
            routes, lengths = balance_routes(routes, lengths)
            ant_solutions.append((max(lengths), sum(lengths), routes, lengths))
            # Update best
            if max(lengths) < best_max or (max(lengths) == best_max and sum(lengths) < best_total):
                best_max = max(lengths)
                best_total = sum(lengths)
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
        # Pheromone update: evaporate
        tau *= (1 - evaporation)
        # Deposit on best-so-far solution
        for route in best_routes:
            if len(route) >= 2:
                for i in range(len(route)-1):
                    tau[route[i], route[i+1]] += evaporation * 0.1  # deposit amount
        # Also deposit on iteration best if different
        iteration_best = min(ant_solutions, key=lambda x: (x[0], x[1]))
        if iteration_best[0] < best_max or (iteration_best[0] == best_max and iteration_best[1] < best_total):
            # Already updated as best above, so skip to avoid double
            pass
        else:
            # deposit on iteration best if no improvement
            for route in iteration_best[2]:
                if len(route) >= 2:
                    for i in range(len(route)-1):
                        tau[route[i], route[i+1]] += evaporation * 0.05
    return best_routes
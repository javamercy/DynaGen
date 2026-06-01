import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Construction: nearest-neighbor TSP + DP minimax split
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current][v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best

    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0][tour[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[tour[r - 2]][tour[r - 1]]
            if r == l + 1:
                seg_dist[l][r] = distance_matrix[0][tour[l]] + distance_matrix[tour[l]][0]
            else:
                seg_dist[l][r] = acc + distance_matrix[tour[r - 1]][0]

    dp = [[math.inf] * (truck_count + 1) for _ in range(m + 1)]
    choice = [[-1] * (truck_count + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(1, m + 1):
        for t in range(1, min(i, truck_count) + 1):
            best_val = math.inf
            best_j = -1
            for j in range(t - 1, i):
                if dp[j][t - 1] < math.inf:
                    cand = max(dp[j][t - 1], seg_dist[j][i])
                    if cand < best_val or (cand == best_val and j < best_j):
                        best_val = cand
                        best_j = j
            dp[i][t] = best_val
            choice[i][t] = best_j

    routes = []
    i = m
    t = truck_count
    while t > 0:
        j = choice[i][t]
        seg = tour[j:i]
        routes.append([0] + seg + [0])
        i = j
        t -= 1
    routes.reverse()
    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def compute_max():
        return max(route_dist(r) for r in routes)

    def copy_routes():
        return [list(r) for r in routes]

    best_routes = copy_routes()
    best_max = compute_max()
    report_best_vrp(best_routes)

    # Local search operators
    def op_2opt():
        nonlocal routes, best_max, best_routes
        for ri, route in enumerate(routes):
            if len(route) <= 3:
                continue
            best_local_dist = route_dist(route)
            best_local_route = route[:]
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_local_dist - 1e-9:
                        best_local_dist = new_dist
                        best_local_route = new_route
                        improved = True
            if improved:
                routes[ri] = best_local_route
                new_max = compute_max()
                if new_max < best_max:
                    best_max = new_max
                    best_routes = copy_routes()
                    report_best_vrp(best_routes)
                return True
        return False

    def op_relocate():
        nonlocal routes, best_max, best_routes
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: dists[i])
        src_route = routes[longest_idx]
        if len(src_route) <= 2:
            return False
        for pos_i in range(1, len(src_route) - 1):
            cust = src_route[pos_i]
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos_j in range(1, len(dst_route)):
                    new_src = src_route[:pos_i] + src_route[pos_i+1:]
                    new_dst = dst_route[:pos_j] + [cust] + dst_route[pos_j:]
                    new_dists = [route_dist(r) for ri, r in enumerate(routes) if ri not in (longest_idx, dst_idx)]
                    new_dist_src = route_dist(new_src)
                    new_dist_dst = route_dist(new_dst)
                    new_max = max([new_dist_src, new_dist_dst] + new_dists)
                    if new_max < compute_max() - 1e-9:
                        routes[longest_idx] = new_src
                        routes[dst_idx] = new_dst
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = copy_routes()
                            report_best_vrp(best_routes)
                        return True
        return False

    def op_swap():
        nonlocal routes, best_max, best_routes
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: dists[i])
        src_route = routes[longest_idx]
        if len(src_route) <= 2:
            return False
        for pos_i in range(1, len(src_route) - 1):
            cust_i = src_route[pos_i]
            for dst_idx in range(len(routes)):
                if dst_idx == longest_idx:
                    continue
                dst_route = routes[dst_idx]
                if len(dst_route) <= 2:
                    continue
                for pos_j in range(1, len(dst_route) - 1):
                    cust_j = dst_route[pos_j]
                    new_src = src_route[:pos_i] + [cust_j] + src_route[pos_i+1:]
                    new_dst = dst_route[:pos_j] + [cust_i] + dst_route[pos_j+1:]
                    new_dists = [route_dist(r) for ri, r in enumerate(routes) if ri not in (longest_idx, dst_idx)]
                    new_dist_src = route_dist(new_src)
                    new_dist_dst = route_dist(new_dst)
                    new_max = max([new_dist_src, new_dist_dst] + new_dists)
                    if new_max < compute_max() - 1e-9:
                        routes[longest_idx] = new_src
                        routes[dst_idx] = new_dst
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = copy_routes()
                            report_best_vrp(best_routes)
                        return True
        return False

    operators = [op_2opt, op_relocate, op_swap]
    num_ops = len(operators)
    rewards = [0.0] * num_ops
    max_iter_ls = 100 * n
    no_improve_limit = 10 * n
    no_improve_count = 0
    T_start = 10.0
    T_end = 0.1
    ls_phase = True
    alns_iter = 0
    max_alns_iter = 50 * n

    # ALNS state
    current_routes = copy_routes()
    current_max = compute_max()

    for iteration in range(max_iter_ls + max_alns_iter):
        if ls_phase and no_improve_count >= no_improve_limit:
            # Switch to ALNS phase
            ls_phase = False
            # Initialize ALNS state
            current_routes = copy_routes()
            current_max = compute_max()
            initial_temp = 0.1 * current_max
            final_temp = 0.001
            cooling_rate = (final_temp / initial_temp) ** (1.0 / max_alns_iter)
            temp = initial_temp
            continue

        if ls_phase:
            # Adaptive local search with softmax
            if iteration >= max_iter_ls:
                break
            temp = T_start + (T_end - T_start) * (iteration / max_iter_ls)
            exp_rewards = [math.exp(r / temp) for r in rewards]
            sum_exp = sum(exp_rewards)
            probs = [e / sum_exp for e in exp_rewards]
            r = random.random()
            cumulative = 0.0
            for op_idx, prob in enumerate(probs):
                cumulative += prob
                if r < cumulative:
                    break
            improved = operators[op_idx]()
            if improved:
                rewards[op_idx] += 1.0
                no_improve_count = 0
                if rewards[op_idx] > 10.0:
                    rewards[op_idx] = 10.0
            else:
                rewards[op_idx] -= 0.5
                no_improve_count += 1
                if rewards[op_idx] < -10.0:
                    rewards[op_idx] = -10.0
        else:
            # ALNS phase
            if alns_iter >= max_alns_iter:
                break
            # Destroy: random removal of ~10% customers
            q = max(1, m // 10)
            removed = []
            new_routes = copy_routes()
            all_cust = [c for route in new_routes for c in route if c != 0]
            random.shuffle(all_cust)
            for c in all_cust[:q]:
                for route in new_routes:
                    if c in route:
                        route.remove(c)
                        removed.append(c)
                        break
            # Repair: greedy insertion minimizing resulting max distance
            random.shuffle(removed)
            for c in removed:
                best_inc = math.inf
                best_ri = -1
                best_pos = -1
                for ri, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [c] + route[pos:]
                        new_dist = route_dist(new_route)
                        other_dists = [route_dist(r) for ri2, r in enumerate(new_routes) if ri2 != ri]
                        new_max = max(new_dist, *other_dists)
                        if new_max < best_inc or (new_max == best_inc and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                            best_inc = new_max
                            best_ri = ri
                            best_pos = pos
                new_routes[best_ri].insert(best_pos, c)
            # Evaluate new solution
            new_max = max(route_dist(r) for r in new_routes)
            delta = new_max - current_max
            accepted = False
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_routes = new_routes
                current_max = new_max
                accepted = True
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in new_routes]
                    report_best_vrp(best_routes)
            if accepted:
                no_improve_count = 0
            else:
                no_improve_count += 1
            # Cool temperature
            temp *= cooling_rate
            alns_iter += 1

    return best_routes
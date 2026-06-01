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

    # --- Initial solution: TSP tour + DP split (minimax) ---
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current, v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best

    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0, tour[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[tour[r - 2], tour[r - 1]]
            if r == l + 1:
                seg_dist[l][r] = distance_matrix[0, tour[l]] + distance_matrix[tour[l], 0]
            else:
                seg_dist[l][r] = acc + distance_matrix[tour[r - 1], 0]

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
        return sum(distance_matrix[route[k], route[k+1]] for k in range(len(route)-1))

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    current = copy_routes(routes)
    current_max = compute_max(current)
    best = copy_routes(current)
    best_max = current_max
    report_best_vrp(best)

    # --- ALNS parameters ---
    max_iter = 2000
    initial_temp = 0.1 * current_max
    final_temp = 0.001
    if initial_temp == 0:
        initial_temp = 0.1
    if final_temp == 0:
        final_temp = 0.001
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max_iter) if initial_temp > 0 else 1.0
    temp = initial_temp

    destroy_ops = ["random", "worst"]
    repair_ops = ["greedy", "regret2"]
    weights_d = [1.0, 1.0]
    weights_r = [1.0, 1.0]
    scores = [0.0, 0.0, 0.0, 0.0]  # random-greedy, random-regret2, worst-greedy, worst-regret2
    usage = [0, 0, 0, 0]
    n_cust = m

    for it in range(max_iter):
        # Select destroy and repair
        d_idx = random.choices(range(2), weights=weights_d)[0]
        r_idx = random.choices(range(2), weights=weights_r)[0]
        op_idx = d_idx * 2 + r_idx

        q = max(1, n_cust // 10)

        # Destroy
        removed = []
        new_routes = copy_routes(current)
        if destroy_ops[d_idx] == "random":
            all_cust = [c for route in new_routes for c in route if c != 0]
            random.shuffle(all_cust)
            for c in all_cust[:q]:
                for route in new_routes:
                    if c in route:
                        route.remove(c)
                        removed.append(c)
                        break
        else:  # worst removal based on detour
            detour = {}
            for route in new_routes:
                for p in range(1, len(route)-1):
                    c = route[p]
                    prev = route[p-1]
                    nxt = route[p+1]
                    det = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    detour[c] = det
            sorted_cust = sorted(detour.items(), key=lambda x: -x[1])
            for c, _ in sorted_cust[:q]:
                for route in new_routes:
                    if c in route:
                        route.remove(c)
                        removed.append(c)
                        break

        # Repair
        random.shuffle(removed)
        if repair_ops[r_idx] == "greedy":
            for c in removed:
                best_max_after = math.inf
                best_ri = -1
                best_pos = -1
                for ri, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [c] + route[pos:]
                        new_dist = route_dist(new_route)
                        other_dists = [route_dist(r) for ri2, r in enumerate(new_routes) if ri2 != ri]
                        new_max = max(new_dist, *other_dists)
                        if new_max < best_max_after or (new_max == best_max_after and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                            best_max_after = new_max
                            best_ri = ri
                            best_pos = pos
                new_routes[best_ri].insert(best_pos, c)
        else:  # regret-2 insertion
            for _ in range(len(removed)):
                best_c = -1
                best_regret = -1
                best_ri = -1
                best_pos = -1
                best_max_val = math.inf
                for c in removed:
                    first = (math.inf, -1, -1)
                    second = (math.inf, -1, -1)
                    for ri, route in enumerate(new_routes):
                        for pos in range(1, len(route)):
                            new_route = route[:pos] + [c] + route[pos:]
                            new_dist = route_dist(new_route)
                            other_dists = [route_dist(r) for ri2, r in enumerate(new_routes) if ri2 != ri]
                            new_max = max(new_dist, *other_dists)
                            if new_max < first[0]:
                                second = first
                                first = (new_max, ri, pos)
                            elif new_max < second[0]:
                                second = (new_max, ri, pos)
                    if first[0] == math.inf:
                        continue
                    regret = second[0] - first[0]
                    if regret > best_regret or (regret == best_regret and c < best_c):
                        best_regret = regret
                        best_c = c
                        best_ri = first[1]
                        best_pos = first[2]
                        best_max_val = first[0]
                if best_c != -1:
                    new_routes[best_ri].insert(best_pos, best_c)
                    removed.remove(best_c)

        # Evaluate
        new_max = compute_max(new_routes)
        delta = new_max - current_max
        accepted = False
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_routes
            current_max = new_max
            accepted = True
            if new_max < best_max:
                best = copy_routes(new_routes)
                best_max = new_max
                report_best_vrp(best)

        # Update scores
        if accepted:
            if new_max < best_max:
                scores[op_idx] += 1.0
            else:
                scores[op_idx] += 0.5
        else:
            scores[op_idx] += 0.0
        usage[op_idx] += 1

        if (it + 1) % 100 == 0:
            for ii in range(4):
                if usage[ii] > 0:
                    scores[ii] /= usage[ii]
            for d in range(2):
                avg = (scores[d*2] + scores[d*2+1]) / 2.0
                weights_d[d] = max(0.1, weights_d[d] * 0.9 + avg * 0.1)
            for r in range(2):
                avg = (scores[r] + scores[2+r]) / 2.0
                weights_r[r] = max(0.1, weights_r[r] * 0.9 + avg * 0.1)
            for ii in range(4):
                scores[ii] = 0.0
                usage[ii] = 0

        temp *= cooling_rate

    # Post-optimization: relocate from longest route using regret-2
    for _ in range(n):
        dists = [route_dist(r) for r in best]
        max_val = max(dists)
        max_idx = dists.index(max_val)
        max_route = best[max_idx]
        improved = False
        best_move = None
        best_new_max = max_val
        for cust in max_route[1:-1]:
            new_max_route = [x for x in max_route if x != cust]
            new_dist_max = route_dist(new_max_route)
            first = (math.inf, -1, -1)
            second = (math.inf, -1, -1)
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = best[other_idx]
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_dist_other = route_dist(new_other)
                    candidate_dists = [new_dist_max if i == max_idx else (new_dist_other if i == other_idx else dists[i]) for i in range(truck_count)]
                    cand_max = max(candidate_dists)
                    if cand_max < first[0]:
                        second = first
                        first = (cand_max, other_idx, pos)
                    elif cand_max < second[0]:
                        second = (cand_max, other_idx, pos)
            if first[0] == math.inf:
                continue
            regret = second[0] - first[0]
            if best_move is None or first[0] < best_new_max:
                best_new_max = first[0]
                best_regret = regret
                best_move = (cust, first[1], first[2])
            elif first[0] == best_new_max:
                tie = False
                if regret > best_regret:
                    tie = True
                elif regret == best_regret:
                    if cust < best_move[0]:
                        tie = True
                    elif cust == best_move[0] and first[1] < best_move[1]:
                        tie = True
                    elif cust == best_move[0] and first[1] == best_move[1] and first[2] < best_move[2]:
                        tie = True
                if tie:
                    best_new_max = first[0]
                    best_regret = regret
                    best_move = (cust, first[1], first[2])
        if best_move is not None and best_new_max < max_val:
            cust, other_idx, pos = best_move
            best[max_idx] = [x for x in max_route if x != cust]
            best[other_idx] = best[other_idx][:pos] + [cust] + best[other_idx][pos:]
            best_max = best_new_max
            report_best_vrp(best)
            improved = True
        if not improved:
            break

    # Final 2-opt on each route
    for ri in range(truck_count):
        route = best[ri]
        if len(route) <= 3:
            continue
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_dist(route) - 1e-9:
                        route[:] = new_route
                        improved = True
                        break
                if improved:
                    break
    new_max = compute_max(best)
    if new_max < best_max:
        best_max = new_max
        report_best_vrp(best)

    return best
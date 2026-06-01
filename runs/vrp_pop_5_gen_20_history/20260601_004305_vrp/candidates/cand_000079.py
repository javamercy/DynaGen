import numpy as np
import math
import heapq

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # TSP tour using nearest neighbor (deterministic)
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

    # DP minimax split (minimize max route distance)
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
        total = 0
        for a in range(len(route) - 1):
            total += distance_matrix[route[a], route[a + 1]]
        return total

    def compute_max():
        return max(route_dist(r) for r in routes)

    best_routes = [list(r) for r in routes]
    best_max = compute_max()
    report_best_vrp(best_routes)

    # Improvement: adaptive regret relocation from max route
    max_iter = m
    regret_k = 3  # start with k=3
    for iteration in range(max_iter):
        dists = [route_dist(r) for r in routes]
        max_val = max(dists)
        max_idx = dists.index(max_val)
        max_route = routes[max_idx]

        revised = False
        best_new_max = max_val
        best_move = None  # (cust, other_idx, pos, new_max_route, new_other_route, regret)

        # Evaluate each customer in the max route
        for cust in max_route[1:-1]:
            temp_route_max = [x for x in max_route if x != cust]
            if len(temp_route_max) < 2:
                continue
            new_dist_max = route_dist(temp_route_max)

            # Collect all candidate insertions for this customer
            candidates = []  # (cand_max, other_idx, pos)
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    new_other_route = other_route[:pos] + [cust] + other_route[pos:]
                    new_dist_other = route_dist(new_other_route)
                    # compute resulting max distance after removal and insertion
                    candidate_dists = [new_dist_max if i == max_idx else (new_dist_other if i == other_idx else dists[i]) for i in range(truck_count)]
                    cand_max = max(candidate_dists)
                    candidates.append((cand_max, other_idx, pos))

            if not candidates:
                continue

            # Sort by cand_max, then by other_idx, then pos for determinism
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))

            # Compute best and regret for current k
            best_cand = candidates[0]
            k_use = min(regret_k, len(candidates))
            if k_use >= 2:
                # regret = sum_{i=1}^{k} (cand_i - cand_0) = sum_{i=1}^{k-1} cand_i - (k-1)*cand_0
                sum_cand = sum(c[0] for c in candidates[1:k_use])
                regret = sum_cand - (k_use - 1) * best_cand[0]
            else:
                regret = 0

            # Compare with current best move
            if best_cand[0] < best_new_max:
                best_new_max = best_cand[0]
                other_idx, pos = best_cand[1], best_cand[2]
                new_other_route = routes[other_idx][:pos] + [cust] + routes[other_idx][pos:]
                best_move = (cust, other_idx, pos, temp_route_max, new_other_route, regret)
            elif best_cand[0] == best_new_max:
                if best_move is None:
                    other_idx, pos = best_cand[1], best_cand[2]
                    new_other_route = routes[other_idx][:pos] + [cust] + routes[other_idx][pos:]
                    best_move = (cust, other_idx, pos, temp_route_max, new_other_route, regret)
                else:
                    # tie-break: higher regret, smaller customer index, smaller other route index, smaller position
                    prev_cand = candidates[0]  # for comparison, but we use stored regret
                    # compare regrets
                    if regret > best_move[5]:
                        other_idx, pos = best_cand[1], best_cand[2]
                        new_other_route = routes[other_idx][:pos] + [cust] + routes[other_idx][pos:]
                        best_move = (cust, other_idx, pos, temp_route_max, new_other_route, regret)
                    elif regret == best_move[5]:
                        if cust < best_move[0]:
                            other_idx, pos = best_cand[1], best_cand[2]
                            new_other_route = routes[other_idx][:pos] + [cust] + routes[other_idx][pos:]
                            best_move = (cust, other_idx, pos, temp_route_max, new_other_route, regret)
                        elif cust == best_move[0]:
                            if best_cand[1] < best_move[1]:
                                other_idx, pos = best_cand[1], best_cand[2]
                                new_other_route = routes[other_idx][:pos] + [cust] + routes[other_idx][pos:]
                                best_move = (cust, other_idx, pos, temp_route_max, new_other_route, regret)
                            elif best_cand[1] == best_move[1] and best_cand[2] < best_move[2]:
                                other_idx, pos = best_cand[1], best_cand[2]
                                new_other_route = routes[other_idx][:pos] + [cust] + routes[other_idx][pos:]
                                best_move = (cust, other_idx, pos, temp_route_max, new_other_route, regret)

        if best_move is not None and best_new_max < best_max:
            cust, other_idx, pos, new_max_route, new_other_route, _ = best_move
            routes[max_idx] = new_max_route
            routes[other_idx] = new_other_route
            best_max = best_new_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            revised = True
        else:
            # No improving move found, reduce regret depth if possible
            if regret_k > 2:
                regret_k -= 1
                revised = True  # to continue loop even though no move

        if not revised:
            break

    return best_routes
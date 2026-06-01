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
        report_best_vrp(routes)
        return routes

    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    # DP minimax split of a permutation into truck_count segments
    def split_permutation(perm):
        # segment distances: seg_dist[l][r] = distance from 0 to perm[l..r-1] back to 0
        seg_dist = [[0]*(m+1) for _ in range(m)]
        for l in range(m):
            acc = distance_matrix[0][perm[l]]
            for r in range(l+1, m+1):
                if r > l+1:
                    acc += distance_matrix[perm[r-2]][perm[r-1]]
                if r == l+1:
                    seg_dist[l][r] = distance_matrix[0][perm[l]] + distance_matrix[perm[l]][0]
                else:
                    seg_dist[l][r] = acc + distance_matrix[perm[r-1]][0]
        dp = [[math.inf]*(truck_count+1) for _ in range(m+1)]
        choice = [[-1]*(truck_count+1) for _ in range(m+1)]
        dp[0][0] = 0
        for i in range(1, m+1):
            for t in range(1, min(i, truck_count)+1):
                best_val = math.inf
                best_j = -1
                for j in range(t-1, i):
                    if dp[j][t-1] < math.inf:
                        cand = max(dp[j][t-1], seg_dist[j][i])
                        if cand < best_val:
                            best_val = cand
                            best_j = j
                dp[i][t] = best_val
                choice[i][t] = best_j
        routes = []
        i = m
        t = truck_count
        while t > 0:
            j = choice[i][t]
            seg = perm[j:i]
            routes.append([0] + seg + [0])
            i = j
            t -= 1
        routes.reverse()
        while len(routes) < truck_count:
            routes.append([0,0])
        return routes

    # Local search: VND with 2-opt, relocate, swap
    def apply_vnd(routes):
        improved = True
        while improved:
            improved = False
            # 2-opt intra-route
            for ri, route in enumerate(routes):
                if len(route) <= 3:
                    continue
                best_dist = route_dist(route)
                best_route = route[:]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_dist(new_route)
                        if new_dist < best_dist - 1e-9:
                            best_dist = new_dist
                            best_route = new_route
                            improved = True
                if improved:
                    routes[ri] = best_route
                    break
            if improved:
                continue
            # relocate from longest route
            dists = [route_dist(r) for r in routes]
            longest_idx = max(range(truck_count), key=lambda i: dists[i])
            src_route = routes[longest_idx]
            if len(src_route) > 2:
                for pos_i in range(1, len(src_route)-1):
                    cust = src_route[pos_i]
                    for dst_idx in range(truck_count):
                        if dst_idx == longest_idx:
                            continue
                        dst_route = routes[dst_idx]
                        for pos_j in range(1, len(dst_route)):
                            new_src = src_route[:pos_i] + src_route[pos_i+1:]
                            new_dst = dst_route[:pos_j] + [cust] + dst_route[pos_j:]
                            new_dists = [route_dist(r) for ri, r in enumerate(routes) if ri not in (longest_idx, dst_idx)]
                            new_max = max(route_dist(new_src), route_dist(new_dst), *new_dists)
                            if new_max < compute_max(routes) - 1e-9:
                                routes[longest_idx] = new_src
                                routes[dst_idx] = new_dst
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue
            # swap between routes
            dists = [route_dist(r) for r in routes]
            sorted_indices = sorted(range(truck_count), key=lambda i: -dists[i])
            for i in range(truck_count):
                for j in range(i+1, truck_count):
                    ri = sorted_indices[i]
                    rj = sorted_indices[j]
                    route_i = routes[ri]
                    route_j = routes[rj]
                    if len(route_i) <= 2 or len(route_j) <= 2:
                        continue
                    for pos_i in range(1, len(route_i)-1):
                        cust_i = route_i[pos_i]
                        for pos_j in range(1, len(route_j)-1):
                            cust_j = route_j[pos_j]
                            new_i = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
                            new_j = route_j[:pos_j] + [cust_i] + route_j[pos_j+1:]
                            new_dists = [route_dist(r) for ri2, r in enumerate(routes) if ri2 not in (ri, rj)]
                            new_max = max(route_dist(new_i), route_dist(new_j), *new_dists)
                            if new_max < compute_max(routes) - 1e-9:
                                routes[ri] = new_i
                                routes[rj] = new_j
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # ACO parameters
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    Q = 1.0
    ant_count = 10
    max_iter = 50
    # pheromone initialization
    tau = np.ones((n, n)) * (1.0 / (n * truck_count))
    # heuristic matrix: 1/(dist+epsilon)
    epsilon = 1e-10
    eta = 1.0 / (distance_matrix + epsilon)
    # set diagonal to 0
    np.fill_diagonal(tau, 0.0)
    np.fill_diagonal(eta, 0.0)

    best_routes = None
    best_max = math.inf

    for iteration in range(max_iter):
        for ant in range(ant_count):
            # construct tour
            visited = [False]*n
            visited[0] = True
            current = 0
            tour = []
            for _ in range(m):
                # compute probabilities for unvisited customers
                unvisited = [c for c in customers if not visited[c]]
                if not unvisited:
                    break
                probs = []
                for c in unvisited:
                    p = (tau[current][c] ** alpha) * (eta[current][c] ** beta)
                    if p < 1e-15:
                        p = 1e-15
                    probs.append(p)
                total = sum(probs)
                # roulette wheel
                r = random.random() * total
                cumulative = 0.0
                selected = unvisited[0]
                for idx, prob in enumerate(probs):
                    cumulative += prob
                    if r <= cumulative:
                        selected = unvisited[idx]
                        break
                tour.append(selected)
                visited[selected] = True
                current = selected
            # split into routes
            routes = split_permutation(tour)
            # apply local search
            routes = apply_vnd(routes)
            cur_max = compute_max(routes)
            if cur_max < best_max - 1e-9:
                best_max = cur_max
                best_routes = copy_routes(routes)
                report_best_vrp(best_routes)
        # evaporate
        tau *= (1 - rho)
        # deposit on best tour edges (if best exists)
        if best_routes is not None:
            # construct best tour from best routes (just concatenate customers in order)
            best_tour = []
            for r in best_routes:
                for node in r:
                    if node != 0:
                        best_tour.append(node)
            # deposit on edges of best tour
            deposit = Q / best_max
            # from depot to first customer
            if best_tour:
                tau[0][best_tour[0]] += deposit
                tau[best_tour[0]][0] += deposit
            for k in range(len(best_tour)-1):
                i = best_tour[k]
                j = best_tour[k+1]
                tau[i][j] += deposit
                tau[j][i] += deposit
            # last customer back to depot
            tau[best_tour[-1]][0] += deposit
            tau[0][best_tour[-1]] += deposit
    return best_routes
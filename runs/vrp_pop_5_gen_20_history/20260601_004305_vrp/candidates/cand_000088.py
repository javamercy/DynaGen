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

    # Construction: TSP tour + DP minimax split
    def construct_solution():
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
        return routes

    routes = construct_solution()

    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def compute_max():
        return max(route_dist(r) for r in routes)

    def copy_routes():
        return [list(r) for r in routes]

    best_routes = copy_routes()
    best_max = compute_max()
    report_best_vrp(best_routes)

    # Operator definitions (modify routes in place, return True if overall max improved)
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
    max_iter = 200 * n
    no_improve_limit = 5 * n
    no_improve_count = 0
    ruin_count = 0
    max_ruin = 5

    def compute_max_with_route(routes, ri, new_route):
        max_dist = 0.0
        for i, r in enumerate(routes):
            if i == ri:
                d = route_dist(new_route)
            else:
                d = route_dist(r)
            if d > max_dist:
                max_dist = d
        return max_dist

    def ruin_and_recreate():
        nonlocal routes, best_max, best_routes, no_improve_count, ruin_count
        m_customers = list(range(1, n))
        if len(m_customers) < 3:
            return False
        num_remove = max(1, int(0.3 * len(m_customers)))
        customers_removed = random.sample(m_customers, num_remove)
        for route in routes:
            for cust in customers_removed:
                if cust in route:
                    route.remove(cust)
        for cust in customers_removed:
            best_new_max = math.inf
            best_ri = -1
            best_pos = -1
            for ri, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_max = compute_max_with_route(routes, ri, new_route)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_ri = ri
                        best_pos = pos
            if best_ri != -1:
                routes[best_ri] = routes[best_ri][:best_pos] + [cust] + routes[best_ri][best_pos:]
            else:
                routes[0].insert(len(routes[0])-1, cust)
        new_max = compute_max()
        if new_max < best_max - 1e-9:
            best_max = new_max
            best_routes = copy_routes()
            report_best_vrp(best_routes)
        no_improve_count = 0
        ruin_count += 1
        return True

    for iteration in range(max_iter):
        if no_improve_count >= no_improve_limit and ruin_count < max_ruin:
            ruin_and_recreate()
            continue
        elif no_improve_count >= no_improve_limit:
            break
        # Random operator selection (uniform)
        op_idx = random.randrange(num_ops)
        improved = operators[op_idx]()
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

    return best_routes
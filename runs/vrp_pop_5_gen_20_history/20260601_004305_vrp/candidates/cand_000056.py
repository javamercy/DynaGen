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

    # --- Initial solution: TSP tour + DP minimax split ---
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

    # Helper functions
    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k + 1]] for k in range(len(route) - 1))

    def compute_max():
        return max(route_dist(r) for r in routes)

    def copy_routes():
        return [list(r) for r in routes]

    current_routes = copy_routes()
    current_max = compute_max()
    best_routes = copy_routes()
    best_max = current_max
    report_best_vrp(best_routes)

    # Tabu parameters
    tabu_tenure = 7
    tabu_customer = [0] * n  # 0 means not tabu, else iteration until which it's tabu
    iteration = 0
    max_iter = 100 * n
    no_improve = 0
    max_no_improve = 50 * n

    while iteration < max_iter and no_improve < max_no_improve:
        iteration += 1
        # Identify longest route(s)
        dists = [route_dist(r) for r in current_routes]
        current_max = max(dists)
        longest_idx = max(range(truck_count), key=lambda i: dists[i])
        longest_route = current_routes[longest_idx]
        best_move = None
        best_new_max = math.inf
        best_move_type = None

        # Candidate moves: 2-opt on longest, relocate from longest, swap between longest and others
        # 2-opt
        if len(longest_route) > 3:
            for i in range(1, len(longest_route) - 2):
                for j in range(i + 1, len(longest_route) - 1):
                    new_route = longest_route[:i] + longest_route[i:j+1][::-1] + longest_route[j+1:]
                    new_dist = route_dist(new_route)
                    new_dists = dists[:]
                    new_dists[longest_idx] = new_dist
                    new_max = max(new_dists)
                    if new_max < best_new_max:
                        # No tabu for 2-opt (internal)
                        best_move = (longest_idx, i, j)
                        best_new_max = new_max
                        best_move_type = '2opt'

        # Relocate from longest to other routes
        for pos_i in range(1, len(longest_route) - 1):
            cust = longest_route[pos_i]
            for other_idx in range(truck_count):
                if other_idx == longest_idx:
                    continue
                other_route = current_routes[other_idx]
                for pos_j in range(1, len(other_route)):
                    new_src = longest_route[:pos_i] + longest_route[pos_i+1:]
                    new_dst = other_route[:pos_j] + [cust] + other_route[pos_j:]
                    new_dists = [route_dist(r) for ri, r in enumerate(current_routes) if ri not in (longest_idx, other_idx)]
                    new_dists.append(route_dist(new_src))
                    new_dists.append(route_dist(new_dst))
                    new_max = max(new_dists)
                    # Check tabu: moving customer out of its current route
                    if tabu_customer[cust] <= iteration or new_max < best_max:
                        if new_max < best_new_max:
                            best_move = (longest_idx, pos_i, other_idx, pos_j)
                            best_new_max = new_max
                            best_move_type = 'relocate'

        # Swap between longest and other routes
        for pos_i in range(1, len(longest_route) - 1):
            cust_i = longest_route[pos_i]
            for other_idx in range(truck_count):
                if other_idx == longest_idx:
                    continue
                other_route = current_routes[other_idx]
                if len(other_route) <= 2:
                    continue
                for pos_j in range(1, len(other_route) - 1):
                    cust_j = other_route[pos_j]
                    new_src = longest_route[:pos_i] + [cust_j] + longest_route[pos_i+1:]
                    new_dst = other_route[:pos_j] + [cust_i] + other_route[pos_j+1:]
                    new_dists = [route_dist(r) for ri, r in enumerate(current_routes) if ri not in (longest_idx, other_idx)]
                    new_dists.append(route_dist(new_src))
                    new_dists.append(route_dist(new_dst))
                    new_max = max(new_dists)
                    # Check tabu for both customers
                    if (tabu_customer[cust_i] <= iteration and tabu_customer[cust_j] <= iteration) or new_max < best_max:
                        if new_max < best_new_max:
                            best_move = (longest_idx, pos_i, other_idx, pos_j)
                            best_new_max = new_max
                            best_move_type = 'swap'

        # Apply best move if found and improves
        if best_move is not None and best_new_max < current_max:
            no_improve = 0
            if best_move_type == '2opt':
                i, j = best_move[1], best_move[2]
                current_routes[longest_idx] = current_routes[longest_idx][:i] + current_routes[longest_idx][i:j+1][::-1] + current_routes[longest_idx][j+1:]
            elif best_move_type == 'relocate':
                src_idx, pos_i, dst_idx, pos_j = best_move
                cust = current_routes[src_idx].pop(pos_i)
                current_routes[dst_idx].insert(pos_j, cust)
                # Update tabu
                tabu_customer[cust] = iteration + tabu_tenure
            elif best_move_type == 'swap':
                src_idx, pos_i, dst_idx, pos_j = best_move
                cust_i = current_routes[src_idx][pos_i]
                cust_j = current_routes[dst_idx][pos_j]
                current_routes[src_idx][pos_i], current_routes[dst_idx][pos_j] = cust_j, cust_i
                tabu_customer[cust_i] = iteration + tabu_tenure
                tabu_customer[cust_j] = iteration + tabu_tenure
            new_max = compute_max()
            if new_max < best_max:
                best_max = new_max
                best_routes = copy_routes()
                report_best_vrp(best_routes)
        else:
            no_improve += 1
    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes
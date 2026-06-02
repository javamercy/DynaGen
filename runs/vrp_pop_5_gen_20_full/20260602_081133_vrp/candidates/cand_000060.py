import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    random.seed(0)
    
    def route_distance(route):
        if len(route) < 2:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    
    def compute_max(routes):
        return max(route_distance(r) for r in routes)
    
    def build_initial():
        # Nearest neighbor giant tour from depot (node 0) but starting from first customer
        start = 1
        giant_path = [start]
        remaining = set(customers)
        remaining.remove(start)
        current = start
        while remaining:
            nearest = None
            min_dist = float('inf')
            for c in remaining:
                d = distance_matrix[current][c]
                if d < min_dist or (d == min_dist and c < nearest):
                    min_dist = d
                    nearest = c
            giant_path.append(nearest)
            remaining.remove(nearest)
            current = nearest
        # DP split to form routes minimizing max distance
        L = len(giant_path)
        if L == 0:
            return [[0, 0] for _ in range(truck_count)]
        if L <= truck_count:
            routes = []
            idx = 0
            for t in range(truck_count):
                if idx < L:
                    route = [0, giant_path[idx], 0]
                    idx += 1
                else:
                    route = [0, 0]
                routes.append(route)
            return routes
        dist0_to = [distance_matrix[0][c] for c in giant_path]
        dist_to0 = [distance_matrix[c][0] for c in giant_path]
        edge = [distance_matrix[giant_path[i]][giant_path[i+1]] for i in range(L-1)]
        pref = [0]*L
        for i in range(1, L):
            pref[i] = pref[i-1] + edge[i-1]
        def seg_cost(i, j):
            return dist0_to[i] + (pref[j] - pref[i]) + dist_to0[j]
        INF = 1e18
        dp = [[INF]*(L+1) for _ in range(truck_count+1)]
        prev = [[-1]*(L+1) for _ in range(truck_count+1)]
        dp[0][0] = 0
        for k in range(1, truck_count+1):
            for i in range(k, L+1):
                best_val = INF
                best_j = -1
                for j in range(k-1, i):
                    if dp[k-1][j] < INF:
                        seg = seg_cost(j, i-1)
                        candidate = max(dp[k-1][j], seg)
                        if candidate < best_val:
                            best_val = candidate
                            best_j = j
                dp[k][i] = best_val
                prev[k][i] = best_j
        if dp[truck_count][L] == INF:
            return None
        split_points = []
        k = truck_count
        i = L
        while k > 0 and i > 0:
            j = prev[k][i]
            split_points.append(j)
            i = j
            k -= 1
        split_points.reverse()
        split_points.append(L)
        routes = []
        for r in range(truck_count):
            start_idx = split_points[r]
            end_idx = split_points[r+1]
            rc = giant_path[start_idx:end_idx]
            if not rc:
                route = [0, 0]
            else:
                route = [0] + rc + [0]
            routes.append(route)
        return routes
    
    def local_search(routes):
        best_routes = [list(r) for r in routes]
        best_max = compute_max(best_routes)
        improved = True
        max_iter = n * 3  # finite bound
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            # 2-opt intra-route
            for r_idx in range(truck_count):
                route = best_routes[r_idx]
                if len(route) <= 3:
                    continue
                best_imp = 0
                best_pair = None
                best_new_route = None
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        old_dist = route_distance(route)
                        imp = old_dist - new_dist
                        if imp > best_imp:
                            best_imp = imp
                            best_pair = (r_idx, new_route)
                if best_imp > 0:
                    new_routes = [list(r) for r in best_routes]
                    new_routes[best_pair[0]] = best_pair[1]
                    new_max = compute_max(new_routes)
                    if new_max < best_max:
                        best_routes = new_routes
                        best_max = new_max
                        improved = True
                        try:
                            report_best_vrp(best_routes)
                        except:
                            pass
            # relocate: move a customer from longest route to another
            if improved:
                continue
            dists = [route_distance(r) for r in best_routes]
            longest_idx = max(range(truck_count), key=lambda x: dists[x])
            other_indices = [i for i in range(truck_count) if i != longest_idx]
            if not other_indices:
                continue
            best_imp = 0
            best_move = None
            from_route = best_routes[longest_idx]
            if len(from_route) <= 2:
                continue
            for cust_pos in range(1, len(from_route)-1):
                cust = from_route[cust_pos]
                for to_idx in other_indices:
                    to_route = best_routes[to_idx]
                    for pos in range(1, len(to_route)):
                        new_from = from_route[:cust_pos] + from_route[cust_pos+1:]
                        if len(new_from) == 1:
                            new_from = [0, 0]
                        new_to = to_route[:pos] + [cust] + to_route[pos:]
                        new_routes = [list(r) for r in best_routes]
                        new_routes[longest_idx] = new_from
                        new_routes[to_idx] = new_to
                        new_max = compute_max(new_routes)
                        if new_max < best_max:
                            imp = best_max - new_max
                            if imp > best_imp:
                                best_imp = imp
                                best_move = (new_from, new_to, to_idx)
            if best_move:
                new_routes = [list(r) for r in best_routes]
                new_routes[longest_idx] = best_move[0]
                new_routes[best_move[2]] = best_move[1]
                best_routes = new_routes
                best_max = compute_max(best_routes)
                improved = True
                try:
                    report_best_vrp(best_routes)
                except:
                    pass
            # exchange: swap customers between two routes
            if improved:
                continue
            best_imp = 0
            best_exchange = None
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = best_routes[r1]
                    route2 = best_routes[r2]
                    if len(route1) <= 2 or len(route2) <= 2:
                        continue
                    for p1 in range(1, len(route1)-1):
                        for p2 in range(1, len(route2)-1):
                            new_routes = [list(r) for r in best_routes]
                            new_routes[r1] = route1[:p1] + [route2[p2]] + route1[p1+1:]
                            new_routes[r2] = route2[:p2] + [route1[p1]] + route2[p2+1:]
                            new_max = compute_max(new_routes)
                            if new_max < best_max:
                                imp = best_max - new_max
                                if imp > best_imp:
                                    best_imp = imp
                                    best_exchange = (new_routes)
            if best_exchange:
                best_routes = best_exchange
                best_max = compute_max(best_routes)
                improved = True
                try:
                    report_best_vrp(best_routes)
                except:
                    pass
        return best_routes
    
    # Build initial solution
    routes = build_initial()
    if routes is None:
        # fallback: assign each customer to a separate route
        routes = []
        for i in range(truck_count):
            if i < len(customers):
                routes.append([0, customers[i], 0])
            else:
                routes.append([0, 0])
    best_routes = local_search(routes)
    try:
        report_best_vrp(best_routes)
    except:
        pass
    return best_routes
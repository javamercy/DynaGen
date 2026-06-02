import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    
    random.seed(0)
    
    best_routes = None
    best_max = float('inf')
    
    def route_distance(route):
        total = 0.0
        for a in range(len(route)-1):
            total += distance_matrix[route[a]][route[a+1]]
        return total
    
    max_starts = min(10, n-1)
    for start_cust in range(1, max_starts+1):
        giant_path = []
        current = start_cust
        giant_path.append(current)
        remaining = set(customers)
        remaining.remove(current)
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
        
        L = len(giant_path)
        dist0_to = [distance_matrix[0][c] for c in giant_path]
        dist_to0 = [distance_matrix[c][0] for c in giant_path]
        edge = [distance_matrix[giant_path[i]][giant_path[i+1]] for i in range(L-1)]
        pref = [0] * L
        for i in range(1, L):
            pref[i] = pref[i-1] + edge[i-1]
        
        def seg_cost(i, j):
            return dist0_to[i] + (pref[j] - pref[i]) + dist_to0[j]
        
        INF = 1e18
        dp = [[INF] * (L+1) for _ in range(truck_count+1)]
        prev = [[-1] * (L+1) for _ in range(truck_count+1)]
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
        
        split_points = []
        k = truck_count
        i = L
        while k > 0 and i > 0:
            j = prev[k][i]
            split_points.append(j)
            i = j
            k -= 1
        split_points.reverse()
        if len(split_points) < truck_count:
            split_points = [0] + [L*(r+1)//truck_count for r in range(truck_count)]
        split_points = split_points[:truck_count] + [L]
        
        routes = []
        for r in range(truck_count):
            start = split_points[r]
            end = split_points[r+1]
            route_customers = giant_path[start:end]
            route = [0] + route_customers + [0]
            routes.append(route)
        while len(routes) < truck_count:
            routes.append([0,0])
        
        current_routes = [list(r) for r in routes]
        current_max = max(route_distance(r) for r in current_routes)
        
        def local_search(routes, current_max):
            improved = True
            while improved:
                improved = False
                # Intra-route 2-opt
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            if j - i == 1:
                                continue
                            old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                            new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                            if new < old - 1e-10:
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                new_dist = route_distance(new_route)
                                other_max = max(route_distance(routes[k]) for k in range(truck_count) if k != r_idx)
                                new_max = max(new_dist, other_max)
                                if new_max < current_max - 1e-10:
                                    routes[r_idx] = new_route
                                    current_max = new_max
                                    improved = True
                                    try:
                                        report_best_vrp(routes)
                                    except:
                                        pass
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
                # Inter-route relocate from longest route
                max_dist = 0
                max_idx = 0
                for i, r in enumerate(routes):
                    d = route_distance(r)
                    if d > max_dist:
                        max_dist = d
                        max_idx = i
                route_from = routes[max_idx]
                for cust in route_from[1:-1]:
                    for r_to_idx in range(truck_count):
                        if r_to_idx == max_idx:
                            continue
                        route_to = routes[r_to_idx]
                        for pos in range(1, len(route_to)):
                            new_route_from = [0] + [x for x in route_from[1:-1] if x != cust] + [0]
                            if len(new_route_from) == 1:
                                new_route_from = [0,0]
                            new_route_to = route_to[:pos] + [cust] + route_to[pos:]
                            new_dist_from = route_distance(new_route_from)
                            new_dist_to = route_distance(new_route_to)
                            other_max = 0.0
                            for k in range(truck_count):
                                if k != max_idx and k != r_to_idx:
                                    dk = route_distance(routes[k])
                                    if dk > other_max:
                                        other_max = dk
                            new_max = max(new_dist_from, new_dist_to, other_max)
                            if new_max < current_max - 1e-10:
                                routes[max_idx] = new_route_from
                                routes[r_to_idx] = new_route_to
                                current_max = new_max
                                improved = True
                                try:
                                    report_best_vrp(routes)
                                except:
                                    pass
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
                # Inter-route exchange
                for i in range(truck_count):
                    for j in range(i+1, truck_count):
                        route_i = routes[i]
                        route_j = routes[j]
                        for pos_i in range(1, len(route_i)-1):
                            for pos_j in range(1, len(route_j)-1):
                                cust_i = route_i[pos_i]
                                cust_j = route_j[pos_j]
                                old_i = distance_matrix[route_i[pos_i-1]][cust_i] + distance_matrix[cust_i][route_i[pos_i+1]]
                                old_j = distance_matrix[route_j[pos_j-1]][cust_j] + distance_matrix[cust_j][route_j[pos_j+1]]
                                new_i = distance_matrix[route_i[pos_i-1]][cust_j] + distance_matrix[cust_j][route_i[pos_i+1]]
                                new_j = distance_matrix[route_j[pos_j-1]][cust_i] + distance_matrix[cust_i][route_j[pos_j+1]]
                                gain = (old_i + old_j) - (new_i + new_j)
                                if gain > 1e-10:
                                    len_i = route_distance(route_i)
                                    len_j = route_distance(route_j)
                                    new_len_i = len_i - old_i + new_i
                                    new_len_j = len_j - old_j + new_j
                                    other_max = 0.0
                                    for k in range(truck_count):
                                        if k != i and k != j:
                                            dk = route_distance(routes[k])
                                            if dk > other_max:
                                                other_max = dk
                                    new_max = max(new_len_i, new_len_j, other_max)
                                    if new_max < current_max - 1e-10:
                                        new_route_i = list(route_i)
                                        new_route_j = list(route_j)
                                        new_route_i[pos_i], new_route_j[pos_j] = cust_j, cust_i
                                        routes[i] = new_route_i
                                        routes[j] = new_route_j
                                        current_max = new_max
                                        improved = True
                                        try:
                                            report_best_vrp(routes)
                                        except:
                                            pass
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
                # Cross-route 2-opt*
                for i in range(truck_count):
                    for j in range(i+1, truck_count):
                        route_i = routes[i]
                        route_j = routes[j]
                        if len(route_i) <= 2 or len(route_j) <= 2:
                            continue
                        for p in range(1, len(route_i)-1):
                            for q in range(1, len(route_j)-1):
                                new_route_i = route_i[:p+1] + route_j[q+1:]
                                new_route_j = route_j[:q+1] + route_i[p+1:]
                                if new_route_i[-1] != 0:
                                    new_route_i.append(0)
                                if new_route_j[-1] != 0:
                                    new_route_j.append(0)
                                if new_route_i[-1] != 0 or new_route_j[-1] != 0:
                                    continue
                                len_i = route_distance(route_i)
                                len_j = route_distance(route_j)
                                new_len_i = route_distance(new_route_i)
                                new_len_j = route_distance(new_route_j)
                                other_max = 0.0
                                for k in range(truck_count):
                                    if k != i and k != j:
                                        dk = route_distance(routes[k])
                                        if dk > other_max:
                                            other_max = dk
                                new_max = max(new_len_i, new_len_j, other_max)
                                if new_max < current_max - 1e-10:
                                    routes[i] = new_route_i
                                    routes[j] = new_route_j
                                    current_max = new_max
                                    improved = True
                                    try:
                                        report_best_vrp(routes)
                                    except:
                                        pass
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            return routes, current_max
        
        current_routes, current_max = local_search(current_routes, current_max)
        
        # Shake and local search cycles - increased to 5, removal 40%, 70% from longest
        for shake_cycle in range(5):
            all_custs = [c for c in customers]
            random.shuffle(all_custs)
            num_remove = max(1, int(0.4 * (n-1)))
            route_distances = [route_distance(r) for r in current_routes]
            longest_idx = max(range(truck_count), key=lambda i: route_distances[i])
            longest_custs = [c for c in current_routes[longest_idx][1:-1]]
            removed = []
            num_from_longest = min(len(longest_custs), int(num_remove * 0.7))
            random.shuffle(longest_custs)
            removed.extend(longest_custs[:num_from_longest])
            remaining_custs = [c for c in all_custs if c not in removed]
            random.shuffle(remaining_custs)
            needed = num_remove - len(removed)
            if needed > 0:
                removed.extend(remaining_custs[:needed])
            for r in current_routes:
                r[:] = [0] + [c for c in r[1:-1] if c not in removed] + [0]
            random.shuffle(removed)
            for cust in removed:
                best_pos = None
                best_route_idx = None
                best_new_max = float('inf')
                for r_idx in range(truck_count):
                    route = current_routes[r_idx]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_distance(new_route)
                        other_max = max(route_distance(current_routes[k]) for k in range(truck_count) if k != r_idx)
                        new_max = max(new_dist, other_max)
                        if new_max < best_new_max - 1e-10 or (abs(new_max - best_new_max) < 1e-10 and cust < (current_routes[best_route_idx][best_pos] if best_pos is not None else float('inf'))):
                            best_new_max = new_max
                            best_route_idx = r_idx
                            best_pos = pos
                if best_route_idx is not None:
                    route = current_routes[best_route_idx]
                    current_routes[best_route_idx] = route[:best_pos] + [cust] + route[best_pos:]
            current_max = max(route_distance(r) for r in current_routes)
            current_routes, current_max = local_search(current_routes, current_max)
            if current_max < best_max:
                best_max = current_max
                best_routes = [list(r) for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except:
                    pass
        
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in current_routes]
            try:
                report_best_vrp(best_routes)
            except:
                pass
    
    if best_routes is None:
        best_routes = [[0,0] for _ in range(truck_count)]
    used = set()
    for r in best_routes:
        for c in r:
            if c != 0:
                used.add(c)
    missing = [c for c in customers if c not in used]
    if missing:
        best_routes[0] = [0] + best_routes[0][1:-1] + missing + [0]
    while len(best_routes) < truck_count:
        best_routes.append([0,0])
    return best_routes
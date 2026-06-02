import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    m = n - 1
    if m == 0:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= m:
        routes = []
        for cust in range(1, n):
            routes.append([0, cust, 0])
        while len(routes) < truck_count:
            routes.append([0, 0])
        best_max = max(sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1)) for route in routes if len(route) > 2)
        for r_idx in range(truck_count):
            if len(routes[r_idx]) > 2:
                route = routes[r_idx]
                current_dist = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
                improved = True
                iter_count = 0
                while improved and iter_count < 100:
                    improved = False
                    iter_count += 1
                    found = False
                    for seg_len in range(1, min(4, len(route)-2)):
                        if found:
                            break
                        for i in range(1, len(route)-seg_len):
                            if found:
                                break
                            segment = route[i:i+seg_len]
                            new_route_no_seg = route[:i] + route[i+seg_len:]
                            for j in range(1, len(new_route_no_seg)):
                                candidate = new_route_no_seg[:j] + segment + new_route_no_seg[j:]
                                new_dist = sum(distance_matrix[candidate[k], candidate[k+1]] for k in range(len(candidate)-1))
                                if new_dist < current_dist:
                                    routes[r_idx] = candidate
                                    current_dist = new_dist
                                    improved = True
                                    found = True
                                    new_max = max(current_dist, max(sum(distance_matrix[route2[i], route2[i+1]] for i in range(len(route2)-1)) for route2 in routes if route2 is not routes[r_idx]))
                                    if new_max < best_max:
                                        best_max = new_max
                                        report_best_vrp(routes)
                                    break
        return routes
    else:
        unvisited = set(range(1, n))
        current = 0
        tour = [0]
        while unvisited:
            best_dist = float('inf')
            best_node = -1
            for node in unvisited:
                d = distance_matrix[current][node]
                if d < best_dist or (d == best_dist and node < best_node):
                    best_dist = d
                    best_node = node
            tour.append(best_node)
            unvisited.remove(best_node)
            current = best_node
        tour.append(0)
        custs = tour[1:-1]
        m = len(custs)
        d_depot_to = [distance_matrix[0][c] for c in custs]
        d_to_depot = [distance_matrix[c][0] for c in custs]
        edge_between = [distance_matrix[custs[i]][custs[i+1]] for i in range(m-1)] if m > 1 else []
        prefix = [0] * m
        for i in range(1, m):
            prefix[i] = prefix[i-1] + edge_between[i-1]
        def seg_dist(l, r):
            return d_depot_to[l] + (prefix[r] - prefix[l]) + d_to_depot[r]
        K = truck_count
        dp = [[float('inf')] * (K+1) for _ in range(m)]
        best_split = [[-1] * (K+1) for _ in range(m)]
        for i in range(m):
            dp[i][1] = seg_dist(0, i)
            best_split[i][1] = 0
        for k in range(2, K+1):
            for i in range(m):
                for j in range(0, i):
                    val = max(dp[j][k-1], seg_dist(j+1, i))
                    if val < dp[i][k]:
                        dp[i][k] = val
                        best_split[i][k] = j+1
        segments = []
        end = m-1
        for k in range(K, 0, -1):
            start = best_split[end][k] if k > 1 else 0
            segments.append((start, end))
            end = start - 1
        segments.reverse()
        routes = []
        for s, e in segments:
            route = [0] + custs[s:e+1] + [0]
            routes.append(route)
        best_max = max(sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1)) for route in routes)
        report_best_vrp(routes)
        for r_idx in range(truck_count):
            route = routes[r_idx]
            current_dist = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
            improved = True
            iter_count = 0
            while improved and iter_count < 100:
                improved = False
                iter_count += 1
                found = False
                for seg_len in range(1, min(4, len(route)-2)):
                    if found:
                        break
                    for i in range(1, len(route)-seg_len):
                        if found:
                            break
                        segment = route[i:i+seg_len]
                        new_route_no_seg = route[:i] + route[i+seg_len:]
                        for j in range(1, len(new_route_no_seg)):
                            candidate = new_route_no_seg[:j] + segment + new_route_no_seg[j:]
                            new_dist = sum(distance_matrix[candidate[k], candidate[k+1]] for k in range(len(candidate)-1))
                            if new_dist < current_dist:
                                routes[r_idx] = candidate
                                current_dist = new_dist
                                improved = True
                                found = True
                                new_max = max(current_dist, max(sum(distance_matrix[route2[i], route2[i+1]] for i in range(len(route2)-1)) for route2 in routes if route2 is not routes[r_idx]))
                                if new_max < best_max:
                                    best_max = new_max
                                    report_best_vrp(routes)
                                break
        return routes
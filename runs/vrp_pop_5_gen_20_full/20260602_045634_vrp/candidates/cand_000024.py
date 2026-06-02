import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]

    customers = list(range(1, n))
    num_customers = n - 1
    k = min(truck_count, num_customers)

    best_max = float('inf')
    best_routes = None

    # Prepare list of starting seeds: farthest customers from depot
    depot_dists = [distance_matrix[0][c] for c in customers]
    sorted_customers = sorted(customers, key=lambda c: (depot_dists[c-1], -c), reverse=True)
    max_starts = min(5, num_customers)

    for start_num in range(max_starts):
        # ---- Deterministic k-medoids clustering with seed start ----
        medoids = [sorted_customers[start_num]]
        remaining = [c for c in customers if c not in medoids]
        while len(medoids) < k:
            min_dists = []
            for c in remaining:
                d = min(distance_matrix[c][m] for m in medoids)
                min_dists.append((d, -c))
            next_idx = max(range(len(min_dists)), key=lambda i: min_dists[i])
            medoids.append(remaining.pop(next_idx))

        clusters = {m: [] for m in medoids}
        for c in customers:
            nearest = min(medoids, key=lambda m: (distance_matrix[c][m], m))
            clusters[nearest].append(c)

        for _ in range(5):
            new_medoids = []
            for m in medoids:
                cluster = clusters[m]
                if not cluster:
                    new_medoids.append(m)
                    continue
                best_c = min(cluster, key=lambda c: sum(distance_matrix[c][o] for o in cluster))
                new_medoids.append(best_c)
            new_clusters = {m: [] for m in new_medoids}
            for c in customers:
                nearest = min(new_medoids, key=lambda m: (distance_matrix[c][m], m))
                new_clusters[nearest].append(c)
            if set(medoids) == set(new_medoids):
                break
            medoids = new_medoids
            clusters = new_clusters

        cluster_list = [clusters.get(m, []) for m in medoids]
        while len(cluster_list) < truck_count:
            cluster_list.append([])

        # ---- Route construction via TSP for each cluster ----
        def route_distance(route):
            if len(route) <= 1:
                return 0.0
            d = 0.0
            for i in range(len(route) - 1):
                d += distance_matrix[route[i]][route[i+1]]
            return d

        def tsp(cluster):
            if not cluster:
                return [0, 0]
            start = max(cluster, key=lambda c: (distance_matrix[0][c], -c))
            route = [0, start]
            unvisited = set(cluster)
            unvisited.remove(start)
            current = start
            while unvisited:
                next_c = min(unvisited, key=lambda c: (distance_matrix[current][c], c))
                route.append(next_c)
                unvisited.remove(next_c)
                current = next_c
            route.append(0)
            # 2-opt
            improved = True
            iter_limit = len(cluster) * 10
            while improved and iter_limit > 0:
                improved = False
                iter_limit -= 1
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_distance(new_route) < route_distance(route):
                            route = new_route
                            improved = True
            # Or-opt
            improved = True
            iter_limit = len(cluster) * 5
            while improved and iter_limit > 0:
                improved = False
                iter_limit -= 1
                for seg_len in [1, 2]:
                    for i in range(1, len(route) - seg_len):
                        seg = route[i:i+seg_len]
                        rest = route[:i] + route[i+seg_len:]
                        for pos in range(1, len(rest)):
                            new_route = rest[:pos] + seg + rest[pos:]
                            if route_distance(new_route) < route_distance(route):
                                route = new_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            return route

        routes = [tsp(cluster) for cluster in cluster_list]
        # Initial report
        current_max = max(route_distance(r) for r in routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

        # ---- Inter-route improvement ----
        improved = True
        max_iter = num_customers * truck_count * 2
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            dists = [route_distance(r) for r in routes]
            max_dist = max(dists)
            longest_idx = min(i for i, d in enumerate(dists) if d == max_dist)
            longest_route = routes[longest_idx]
            if len(longest_route) <= 2:
                break
            best_move = None
            best_new_max = max_dist
            # Relocate
            for cust_idx in range(1, len(longest_route) - 1):
                cust = longest_route[cust_idx]
                new_long = longest_route[:cust_idx] + longest_route[cust_idx+1:]
                dist_long = route_distance(new_long)
                for other_idx, other_route in enumerate(routes):
                    if other_idx == longest_idx:
                        continue
                    best_other_dist = None
                    best_pos = None
                    for pos in range(1, len(other_route)):
                        new_other = other_route[:pos] + [cust] + other_route[pos:]
                        dist_other = route_distance(new_other)
                        if best_other_dist is None or dist_other < best_other_dist:
                            best_other_dist = dist_other
                            best_pos = pos
                    if best_pos is None:
                        continue
                    other_dist = best_other_dist
                    max_rest = 0.0
                    for j, r in enumerate(routes):
                        if j not in (longest_idx, other_idx):
                            d = route_distance(r)
                            if d > max_rest:
                                max_rest = d
                    new_max = max(dist_long, other_dist, max_rest)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = ('relocate', longest_idx, other_idx, cust_idx, best_pos, cust)
            # If no relocate, try swap
            if best_move is None:
                for other_idx, other_route in enumerate(routes):
                    if other_idx == longest_idx:
                        continue
                    if len(other_route) <= 2:
                        continue
                    for cust_idx in range(1, len(longest_route) - 1):
                        cust = longest_route[cust_idx]
                        for other_cust_idx in range(1, len(other_route) - 1):
                            other_cust = other_route[other_cust_idx]
                            new_long = longest_route[:cust_idx] + [other_cust] + longest_route[cust_idx+1:]
                            new_other = other_route[:other_cust_idx] + [cust] + other_route[other_cust_idx+1:]
                            dist_long = route_distance(new_long)
                            dist_other = route_distance(new_other)
                            max_rest = 0.0
                            for j, r in enumerate(routes):
                                if j not in (longest_idx, other_idx):
                                    d = route_distance(r)
                                    if d > max_rest:
                                        max_rest = d
                            new_max = max(dist_long, dist_other, max_rest)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('swap', longest_idx, other_idx, cust_idx, other_cust_idx, cust, other_cust)
            if best_move is not None:
                if best_move[0] == 'relocate':
                    _, li, oi, ci, pos, cust = best_move
                    new_long = routes[li][:ci] + routes[li][ci+1:]
                    new_other = routes[oi][:pos] + [cust] + routes[oi][pos:]
                    routes[li] = new_long
                    routes[oi] = new_other
                else:  # swap
                    _, li, oi, ci, oci, cust, other_cust = best_move
                    new_long = routes[li][:ci] + [other_cust] + routes[li][ci+1:]
                    new_other = routes[oi][:oci] + [cust] + routes[oi][oci+1:]
                    routes[li] = new_long
                    routes[oi] = new_other
                new_max = max(route_distance(r) for r in routes)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                improved = True

    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes
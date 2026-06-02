import numpy as np
import random
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    customers = list(range(1, n))
    num_customers = n - 1
    k = min(truck_count, num_customers)
    
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
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
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
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
                for i in range(1, len(route)-seg_len):
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
    
    def build_routes(medoids):
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
        routes = [tsp(cluster) for cluster in cluster_list]
        # inter-route improvement over all route pairs
        improved = True
        max_iter = num_customers * truck_count * 2
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            dists = [route_distance(r) for r in routes]
            best_move = None
            best_new_max = max(dists)
            # consider all pairs (i, j) with i != j
            for i in range(truck_count):
                if len(routes[i]) <= 2:
                    continue
                for j in range(truck_count):
                    if i == j:
                        continue
                    # relocate: move customer from i to j
                    for cust_idx in range(1, len(routes[i])-1):
                        cust = routes[i][cust_idx]
                        new_route_i = routes[i][:cust_idx] + routes[i][cust_idx+1:]
                        for pos in range(1, len(routes[j])):
                            new_route_j = routes[j][:pos] + [cust] + routes[j][pos:]
                            # compute new max
                            new_dists = dists.copy()
                            new_dists[i] = route_distance(new_route_i)
                            new_dists[j] = route_distance(new_route_j)
                            new_max = max(new_dists)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('relocate', i, j, cust_idx, pos)
                    # swap: exchange customers between i and j
                    if len(routes[j]) > 2:
                        for ci in range(1, len(routes[i])-1):
                            for cj in range(1, len(routes[j])-1):
                                new_route_i = routes[i][:ci] + [routes[j][cj]] + routes[i][ci+1:]
                                new_route_j = routes[j][:cj] + [routes[i][ci]] + routes[j][cj+1:]
                                new_dists = dists.copy()
                                new_dists[i] = route_distance(new_route_i)
                                new_dists[j] = route_distance(new_route_j)
                                new_max = max(new_dists)
                                if new_max < best_new_max:
                                    best_new_max = new_max
                                    best_move = ('swap', i, j, ci, cj)
            if best_move is not None:
                typ, i, j = best_move[0], best_move[1], best_move[2]
                if typ == 'relocate':
                    _, _, _, ci, pos = best_move
                    cust = routes[i][ci]
                    routes[i] = routes[i][:ci] + routes[i][ci+1:]
                    routes[j] = routes[j][:pos] + [cust] + routes[j][pos:]
                else:
                    _, _, _, ci, cj = best_move
                    routes[i], routes[j] = routes[i][:ci] + [routes[j][cj]] + routes[i][ci+1:], routes[j][:cj] + [routes[i][ci]] + routes[j][cj+1:]
                improved = True
        # load balancing: move from longest to shortest repeatedly
        for _ in range(truck_count * 2):
            dists = [route_distance(r) for r in routes]
            max_idx = max(range(truck_count), key=lambda x: (dists[x], -x))
            min_idx = min(range(truck_count), key=lambda x: (dists[x], x))
            if max_idx == min_idx:
                break
            max_route = routes[max_idx]
            min_route = routes[min_idx]
            if len(max_route) <= 2 or len(min_route) == 0:
                break
            best_improve = 0.0
            best_cust_idx = None
            best_pos = None
            for cust_idx in range(1, len(max_route)-1):
                cust = max_route[cust_idx]
                new_max = max_route[:cust_idx] + max_route[cust_idx+1:]
                dist_new_max = route_distance(new_max)
                for pos in range(1, len(min_route)):
                    new_min = min_route[:pos] + [cust] + min_route[pos:]
                    dist_new_min = route_distance(new_min)
                    old_max_dist = dists[max_idx]
                    old_min_dist = dists[min_idx]
                    new_dists = dists.copy()
                    new_dists[max_idx] = dist_new_max
                    new_dists[min_idx] = dist_new_min
                    new_max_dist = max(new_dists)
                    if new_max_dist < max(dists):
                        improve = max(dists) - new_max_dist
                        if improve > best_improve:
                            best_improve = improve
                            best_cust_idx = cust_idx
                            best_pos = pos
            if best_cust_idx is not None:
                cust = max_route[best_cust_idx]
                routes[max_idx] = max_route[:best_cust_idx] + max_route[best_cust_idx+1:]
                routes[min_idx] = min_route[:best_pos] + [cust] + min_route[best_pos:]
        return routes
    
    best_routes = None
    best_max_dist = float('inf')
    num_starts = 10
    for seed in range(num_starts):
        random.seed(seed)
        medoids = random.sample(customers, k) if k <= len(customers) else customers[:]
        routes = build_routes(medoids)
        while len(routes) < truck_count:
            routes.append([0,0])
        max_dist = max(route_distance(r) for r in routes)
        if max_dist < best_max_dist:
            best_max_dist = max_dist
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
    return best_routes
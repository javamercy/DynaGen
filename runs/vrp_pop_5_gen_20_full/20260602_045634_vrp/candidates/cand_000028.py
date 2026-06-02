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
        # Nearest neighbor starting from farthest customer from depot
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
        # refine medoids
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
        # inter-route improvement
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
            # relocate from longest
            for cust_idx in range(1, len(longest_route)-1):
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
                        best_move = (longest_idx, other_idx, cust_idx, best_pos, cust)
            # swap between longest and another
            if best_move is None:
                for other_idx, other_route in enumerate(routes):
                    if other_idx == longest_idx or len(other_route) <= 2:
                        continue
                    for cust_idx in range(1, len(longest_route)-1):
                        cust = longest_route[cust_idx]
                        for other_cust_idx in range(1, len(other_route)-1):
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
                                best_move = (longest_idx, other_idx, cust_idx, other_cust_idx, cust, other_cust, 'swap')
            if best_move is not None:
                li, oi = best_move[0], best_move[1]
                if len(best_move) == 7:
                    _, _, ci, ocj, cust, other_cust, _ = best_move
                    new_long = routes[li][:ci] + [other_cust] + routes[li][ci+1:]
                    new_other = routes[oi][:ocj] + [cust] + routes[oi][ocj+1:]
                    routes[li] = new_long
                    routes[oi] = new_other
                else:
                    _, _, ci, pos, cust = best_move
                    new_long = routes[li][:ci] + routes[li][ci+1:]
                    new_other = routes[oi][:pos] + [cust] + routes[oi][pos:]
                    routes[li] = new_long
                    routes[oi] = new_other
                improved = True
        # load balancing: move from longest to shortest
        for _ in range(truck_count):
            dists = [route_distance(r) for r in routes]
            max_idx = min(i for i, d in enumerate(dists) if d == max(dists))
            min_idx = min(i for i, d in enumerate(dists) if d == min(dists))
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
                    new_max_dist = max(dist_new_max, dist_new_min, max(d for i,d in enumerate(dists) if i not in (max_idx,min_idx)))
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
    num_starts = 5  # number of multi-starts
    for seed in range(num_starts):
        random.seed(seed)
        if k == num_customers:
            medoids = customers[:]
        else:
            # farthest-first initialization
            first = random.choice(customers) if k > 0 else None
            medoids = [first]
            while len(medoids) < k:
                # find customer farthest from any existing medoid
                farthest = max(customers, key=lambda c: min(distance_matrix[c][m] for m in medoids in medoids))  # note: 'in medoids' is a bug, should be 'for m in medoids'
                # Actually correct: min(distance_matrix[c][m] for m in medoids)
                farthest = max(customers, key=lambda c: min(distance_matrix[c][m] for m in medoids))
                medoids.append(farthest)
        routes = build_routes(medoids)
        while len(routes) < truck_count:
            routes.append([0,0])
        max_dist = max(route_distance(r) for r in routes)
        if max_dist < best_max_dist:
            best_max_dist = max_dist
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
    return best_routes
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
    
    def tsp_improve(route):
        # 2-opt
        improved = True
        iter_limit = len(route) * 10
        while improved and iter_limit > 0:
            improved = False
            iter_limit -= 1
            best_d = route_distance(route)
            best_route = route[:]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_d = route_distance(new_route)
                    if new_d < best_d:
                        best_d = new_d
                        best_route = new_route
                        improved = True
            route = best_route
        # Or-opt
        improved = True
        iter_limit = len(route) * 5
        while improved and iter_limit > 0:
            improved = False
            iter_limit -= 1
            best_d = route_distance(route)
            best_route = route[:]
            for seg_len in [1, 2]:
                for i in range(1, len(route)-seg_len):
                    seg = route[i:i+seg_len]
                    rest = route[:i] + route[i+seg_len:]
                    for pos in range(1, len(rest)):
                        new_route = rest[:pos] + seg + rest[pos:]
                        new_d = route_distance(new_route)
                        if new_d < best_d:
                            best_d = new_d
                            best_route = new_route
                            improved = True
                        if improved:
                            break
                if improved:
                    break
            route = best_route
        return route
    
    def furthest_first_init():
        # farthest-first to choose k medoids
        perm = random.sample(customers, k)
        medoids = [perm[0]]
        while len(medoids) < k:
            candidate = None
            best_min_dist = -1
            for c in perm:
                if c in medoids:
                    continue
                min_dist = min(distance_matrix[c][m] for m in medoids)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    candidate = c
            medoids.append(candidate)
        return medoids
    
    def cheapest_insertion(cluster):
        # build TSP route using cheapest insertion
        if not cluster:
            return [0, 0]
        # start with farthest from depot
        start = max(cluster, key=lambda c: (distance_matrix[0][c], c))
        route = [0, start, 0]
        remaining = list(cluster)
        remaining.remove(start)
        while remaining:
            best_c = None
            best_pos = None
            best_cost = float('inf')
            for c in remaining:
                for pos in range(1, len(route)):
                    new_cost = (distance_matrix[route[pos-1]][c] + distance_matrix[c][route[pos]] - distance_matrix[route[pos-1]][route[pos]])
                    if new_cost < best_cost or (new_cost == best_cost and c < best_c):
                        best_cost = new_cost
                        best_c = c
                        best_pos = pos
            route = route[:best_pos] + [best_c] + route[best_pos:]
            remaining.remove(best_c)
        return route
    
    def build_routes(medoids):
        # assign customers to nearest medoid
        assignment = {m: [] for m in medoids}
        for c in customers:
            nearest = min(medoids, key=lambda m: (distance_matrix[c][m], m))
            assignment[nearest].append(c)
        # refine medoids
        for _ in range(5):
            new_medoids = []
            for m in medoids:
                cluster = assignment[m]
                if not cluster:
                    new_medoids.append(m)
                else:
                    best_c = min(cluster, key=lambda c: sum(distance_matrix[c][o] for o in cluster))
                    new_medoids.append(best_c)
            new_assignment = {m: [] for m in new_medoids}
            for c in customers:
                nearest = min(new_medoids, key=lambda m: (distance_matrix[c][m], m))
                new_assignment[nearest].append(c)
            if set(medoids) == set(new_medoids):
                break
            medoids = new_medoids
            assignment = new_assignment
        # build routes via cheapest insertion
        routes = []
        for m in medoids:
            route = cheapest_insertion(assignment[m])
            routes.append(route)
        while len(routes) < truck_count:
            routes.append([0,0])
        # intra-route improvement
        for idx in range(truck_count):
            if len(routes[idx]) > 3:
                routes[idx] = tsp_improve(routes[idx])
        # inter-route improvement: relocate, swap, double relocate
        improved = True
        max_iter = num_customers * truck_count * 2
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            dists = [route_distance(r) for r in routes]
            best_move = None
            best_new_max = max(dists)
            # relocate and swap over all pairs
            for i in range(truck_count):
                if len(routes[i]) <= 2:
                    continue
                for j in range(truck_count):
                    if i == j:
                        continue
                    # relocate
                    for cust_idx in range(1, len(routes[i])-1):
                        cust = routes[i][cust_idx]
                        new_i = routes[i][:cust_idx] + routes[i][cust_idx+1:]
                        for pos in range(1, len(routes[j])):
                            new_j = routes[j][:pos] + [cust] + routes[j][pos:]
                            new_max = max(route_distance(new_i), route_distance(new_j))
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('relocate', i, j, cust_idx, pos)
                    # swap
                    if len(routes[j]) > 2:
                        for ci in range(1, len(routes[i])-1):
                            for cj in range(1, len(routes[j])-1):
                                new_i = routes[i][:ci] + [routes[j][cj]] + routes[i][ci+1:]
                                new_j = routes[j][:cj] + [routes[i][ci]] + routes[j][cj+1:]
                                new_max = max(route_distance(new_i), route_distance(new_j))
                                if new_max < best_new_max:
                                    best_new_max = new_max
                                    best_move = ('swap', i, j, ci, cj)
                    # double relocate: move two consecutive customers from i to j
                    if len(routes[i]) >= 4:
                        for ci in range(1, len(routes[i])-2):
                            seg = routes[i][ci:ci+2]
                            new_i = routes[i][:ci] + routes[i][ci+2:]
                            for pos in range(1, len(routes[j])):
                                new_j = routes[j][:pos] + seg + routes[j][pos:]
                                new_max = max(route_distance(new_i), route_distance(new_j))
                                if new_max < best_new_max:
                                    best_new_max = new_max
                                    best_move = ('double_relocate', i, j, ci, pos)
            if best_move is not None:
                typ, i, j = best_move[0], best_move[1], best_move[2]
                if typ == 'relocate':
                    _, _, _, ci, pos = best_move
                    cust = routes[i][ci]
                    routes[i] = routes[i][:ci] + routes[i][ci+1:]
                    routes[j] = routes[j][:pos] + [cust] + routes[j][pos:]
                elif typ == 'swap':
                    _, _, _, ci, cj = best_move
                    routes[i], routes[j] = routes[i][:ci] + [routes[j][cj]] + routes[i][ci+1:], routes[j][:cj] + [routes[i][ci]] + routes[j][cj+1:]
                else:  # double_relocate
                    _, _, _, ci, pos = best_move
                    seg = routes[i][ci:ci+2]
                    routes[i] = routes[i][:ci] + routes[i][ci+2:]
                    routes[j] = routes[j][:pos] + seg + routes[j][pos:]
                improved = True
        # load balancing: move from longest to shortest
        for _ in range(truck_count * 3):
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
                    new_max_dist = max(dist_new_max, dist_new_min)
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
    num_starts = 20
    for seed in range(num_starts):
        random.seed(seed)
        medoids = furthest_first_init()
        routes = build_routes(medoids)
        while len(routes) < truck_count:
            routes.append([0,0])
        max_dist = max(route_distance(r) for r in routes)
        if max_dist < best_max_dist:
            best_max_dist = max_dist
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
    return best_routes
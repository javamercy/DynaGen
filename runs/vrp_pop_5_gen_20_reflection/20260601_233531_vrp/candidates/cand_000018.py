import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    customers = list(range(1, n))
    # Step 1: Clarke-Wright savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    routes = [[0, c, 0] for c in customers]
    cust_to_route = {c: idx for idx, c in enumerate(customers)}
    
    for s, i, j in savings:
        if cust_to_route[i] == cust_to_route[j]:
            continue
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        route_i = routes[ri]
        route_j = routes[rj]
        endpoints_i = [route_i[1], route_i[-2]]
        endpoints_j = [route_j[1], route_j[-2]]
        if i not in endpoints_i or j not in endpoints_j:
            continue
        if route_i[-2] == i:
            route_i[1:-1] = route_i[-2:0:-1]
        if route_j[1] == j:
            route_j[1:-1] = route_j[-2:0:-1]
        if route_i[1] == i and route_j[-2] == j:
            new_route = route_i[:-1] + route_j[1:]
            routes[ri] = new_route
            routes[rj] = [0, 0]
            for c in route_j[1:-1]:
                cust_to_route[c] = ri
    
    non_empty = [r for r in routes if len(r) > 2]
    empty_count = truck_count - len(non_empty)
    if empty_count < 0:
        non_empty.sort(key=lambda r: -len(r))
        routes = non_empty[:truck_count]
        extra = non_empty[truck_count:]
        for ext in extra:
            shortest_idx = min(range(truck_count), key=lambda i: len(routes[i]))
            for c in ext[1:-1]:
                routes[shortest_idx].insert(-1, c)
    else:
        routes = non_empty + [[0, 0] for _ in range(empty_count)]
    
    while len(routes) < truck_count:
        routes.append([0, 0])
    routes = routes[:truck_count]
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    # Step 2: Intra-route 2-opt
    for idx, route in enumerate(routes):
        if len(route) <= 3:
            continue
        improved = True
        max_iters = len(route) * len(route)
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            best_gain = 0
            best_ij = None
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    gain = old - new
                    if gain > best_gain:
                        best_gain = gain
                        best_ij = (i, j)
            if best_gain > 0:
                i, j = best_ij
                route[i:j+1] = route[i:j+1][::-1]
                improved = True
    
    # Step 3: Inter-route improvement focusing on max route distance, only moving to shortest route
    def total_customers():
        cnt = 0
        for r in routes:
            cnt += len(r) - 2
        return cnt
    
    max_iter = 2 * total_customers()
    for iteration in range(max_iter):
        dists = [route_distance(r) for r in routes]
        max_dist = max(dists)
        max_idx = dists.index(max_dist)
        min_idx = dists.index(min(dists))
        best_improvement = None
        best_new_max = max_dist
        max_route = routes[max_idx]
        cust_list = max_route[1:-1]
        for c in cust_list:
            other_route = routes[min_idx]
            for pos in range(1, len(other_route)):
                new_max_route = [x for x in max_route if x != c]
                new_other_route = other_route[:pos] + [c] + other_route[pos:]
                if len(new_other_route) < 3:
                    new_other_route = [0, c, 0]
                new_max_dist = route_distance(new_max_route)
                new_other_dist = route_distance(new_other_route)
                candidate_max = max(new_max_dist, new_other_dist)
                if candidate_max < best_new_max:
                    best_new_max = candidate_max
                    best_improvement = ('move', c, min_idx, pos, new_max_route, new_other_route)
        if best_improvement is not None:
            _, c, other_idx, pos, new_max, new_other = best_improvement
            routes[max_idx] = new_max
            routes[other_idx] = new_other
            dists = [route_distance(r) for r in routes]
            new_max_dist = max(dists)
            if new_max_dist < max_dist:
                report_best_vrp(routes)
        else:
            break
    
    for r in routes:
        if r[0] != 0:
            r.insert(0, 0)
        if r[-1] != 0:
            r.append(0)
    if len(routes) < truck_count:
        routes.extend([[0, 0]] * (truck_count - len(routes)))
    routes = routes[:truck_count]
    return routes
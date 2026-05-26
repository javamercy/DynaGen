import numpy as np
import random
from typing import List

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    random.seed(0)
    np.random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    best_routes = None
    best_max = float('inf')
    
    for _ in range(10):  # restarts
        perm = customers[:]
        random.shuffle(perm)
        routes = [[0, 0] for _ in range(truck_count)]
        # Greedy construction
        for cust in perm:
            best_truck = None
            best_pos = None
            best_new_max = float('inf')
            best_total = float('inf')
            for t in range(truck_count):
                route = routes[t]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_distance(new_route, distance_matrix)
                    other_dists = [route_distance(routes[tt], distance_matrix) for tt in range(truck_count) if tt != t]
                    cur_max = max(max(other_dists) if other_dists else 0, new_dist)
                    cur_total = sum(other_dists) + new_dist
                    if cur_max < best_new_max or (cur_max == best_new_max and cur_total < best_total):
                        best_new_max = cur_max
                        best_total = cur_total
                        best_truck = t
                        best_pos = pos
            routes[best_truck] = routes[best_truck][:best_pos] + [cust] + routes[best_truck][best_pos:]
        
        # Evaluate initial
        max_dist = max(route_distance(r, distance_matrix) for r in routes)
        if max_dist < best_max - 1e-12:
            best_max = max_dist
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        
        # Local search
        no_improve = 0
        for _ in range(50):  # passes
            improved = False
            max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t], distance_matrix))
            max_route = routes[max_idx]
            max_route_dist = route_distance(max_route, distance_matrix)
            # Try relocate
            for cust in max_route[1:-1]:
                for t in range(truck_count):
                    if t == max_idx: continue
                    other_route = routes[t]
                    for pos in range(1, len(other_route)):
                        new_max_route = max_route[:max_route.index(cust)] + max_route[max_route.index(cust)+1:]
                        new_other_route = other_route[:pos] + [cust] + other_route[pos:]
                        new_max_dist = max(route_distance(new_max_route, distance_matrix),
                                           route_distance(new_other_route, distance_matrix))
                        for tt in range(truck_count):
                            if tt not in (max_idx, t):
                                new_max_dist = max(new_max_dist, route_distance(routes[tt], distance_matrix))
                        if new_max_dist < max_route_dist - 1e-12:
                            routes[max_idx] = new_max_route
                            routes[t] = new_other_route
                            max_route_dist = new_max_dist
                            improved = True
                            break
                    if improved: break
                if improved: break
            if not improved:
                # Try swap
                for t in range(truck_count):
                    if t == max_idx: continue
                    other_route = routes[t]
                    for cust_i in max_route[1:-1]:
                        for cust_j in other_route[1:-1]:
                            new_max_route = max_route[:max_route.index(cust_i)] + [cust_j] + max_route[max_route.index(cust_i)+1:]
                            new_other_route = other_route[:other_route.index(cust_j)] + [cust_i] + other_route[other_route.index(cust_j)+1:]
                            new_max_dist = max(route_distance(new_max_route, distance_matrix),
                                               route_distance(new_other_route, distance_matrix))
                            for tt in range(truck_count):
                                if tt not in (max_idx, t):
                                    new_max_dist = max(new_max_dist, route_distance(routes[tt], distance_matrix))
                            if new_max_dist < max_route_dist - 1e-12:
                                routes[max_idx] = new_max_route
                                routes[t] = new_other_route
                                max_route_dist = new_max_dist
                                improved = True
                                break
                        if improved: break
                    if improved: break
            if not improved:
                # Intra-route 2-opt on max route
                best_impr = 0
                best_i = best_j = None
                for i in range(1, len(max_route)-2):
                    for j in range(i+1, len(max_route)-1):
                        new_route = max_route[:i] + max_route[i:j+1][::-1] + max_route[j+1:]
                        new_dist = route_distance(new_route, distance_matrix)
                        if new_dist < max_route_dist - best_impr:
                            best_impr = max_route_dist - new_dist
                            best_i, best_j = i, j
                if best_impr > 1e-12:
                    new_max_route = max_route[:best_i] + max_route[best_i:best_j+1][::-1] + max_route[best_j+1:]
                    routes[max_idx] = new_max_route
                    max_route_dist = route_distance(new_max_route, distance_matrix)
                    improved = True
            if improved:
                no_improve = 0
                new_max = max(route_distance(r, distance_matrix) for r in routes)
                if new_max < best_max - 1e-12:
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
            else:
                no_improve += 1
                if no_improve >= 10:
                    # Perturbation: eject 1-3 customers from max route and reinsert greedily
                    eject_count = min(3, len(max_route)-2)
                    eject_indices = random.sample(range(1, len(max_route)-1), eject_count)
                    ejected = [max_route[i] for i in sorted(eject_indices, reverse=True)]
                    for i in sorted(eject_indices, reverse=True):
                        max_route.pop(i)
                    for cust in ejected:
                        best_truck = None
                        best_pos = None
                        best_new_max = float('inf')
                        best_total = float('inf')
                        for t in range(truck_count):
                            route = routes[t]
                            for pos in range(1, len(route)):
                                new_route = route[:pos] + [cust] + route[pos:]
                                new_dist = route_distance(new_route, distance_matrix)
                                other_dists = [route_distance(routes[tt], distance_matrix) for tt in range(truck_count) if tt != t]
                                cur_max = max(max(other_dists) if other_dists else 0, new_dist)
                                cur_total = sum(other_dists) + new_dist
                                if cur_max < best_new_max or (cur_max == best_new_max and cur_total < best_total):
                                    best_new_max = cur_max
                                    best_total = cur_total
                                    best_truck = t
                                    best_pos = pos
                        routes[best_truck] = routes[best_truck][:best_pos] + [cust] + routes[best_truck][best_pos:]
                    max_route_dist = max(route_distance(r, distance_matrix) for r in routes)
                    no_improve = 0
            if no_improve >= 20:
                break
    
    return best_routes
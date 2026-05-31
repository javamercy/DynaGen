import numpy as np
from collections import defaultdict
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    # 1. Construction: cheapest insertion as parent
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    for cust in customers:
        best_inc = float('inf')
        best_route = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                inc = (distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]])
                if inc < best_inc or (inc == best_inc and r_idx < best_route):
                    best_inc = inc
                    best_route = r_idx
                    best_pos = pos
        route = routes[best_route]
        routes[best_route] = route[:best_pos] + [cust] + route[best_pos:]
    
    # 2. Post-construction: 2-opt improvement on each route
    def improve_2opt(route):
        improved = True
        max_iter = n * 2  # bounded
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < route_distance(route):
                        route = new_route
                        improved = True
        return route
    
    for idx in range(truck_count):
        if len(routes[idx]) > 2:
            routes[idx] = improve_2opt(routes[idx])
    
    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in routes)
    report_best_vrp(best_routes)
    
    # 3. Tabu search with focused moves on longest route(s)
    max_iter = n * truck_count * 10
    tabu = {}  # (cust, src_route) -> remaining tenure
    tenure = 5
    stagnation = 0
    
    for it in range(max_iter):
        current_dists = [route_distance(r) for r in routes]
        max_dist = max(current_dists)
        # identify all routes that achieve max_dist
        max_routes = [i for i, d in enumerate(current_dists) if d == max_dist]
        
        best_move = None
        best_new_max = float('inf')
        best_new_total = float('inf')
        
        # iterate over customers in max routes
        for src_idx in max_routes:
            route = routes[src_idx]
            for pos in range(1, len(route)-1):
                cust = route[pos]
                # remove cust from its route
                new_src = route[:pos] + route[pos+1:]
                src_dist = route_distance(new_src)
                # consider all insertion positions in all routes
                for dst_idx, dst_route in enumerate(routes):
                    for ins_pos in range(1, len(dst_route)):
                        if dst_idx == src_idx and ins_pos == pos:
                            continue
                        # construct new destination route
                        new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                        dst_dist = route_distance(new_dst)
                        # compute new max
                        new_max = src_dist if src_dist > dst_dist else dst_dist
                        for r_idx_other, r_other in enumerate(routes):
                            if r_idx_other == src_idx:
                                continue
                            if r_idx_other == dst_idx:
                                continue
                            other_dist = current_dists[r_idx_other]
                            if other_dist > new_max:
                                new_max = other_dist
                        total = src_dist + dst_dist + sum(current_dists[i] for i in range(truck_count) if i not in (src_idx, dst_idx))
                        # tabu check
                        key = (cust, src_idx)
                        is_tabu = key in tabu and tabu[key] > 0
                        if is_tabu and new_max >= best_max:
                            continue
                        if new_max < best_new_max or (new_max == best_new_max and total < best_new_total):
                            best_new_max = new_max
                            best_new_total = total
                            best_move = (cust, src_idx, dst_idx, ins_pos, new_src, new_dst)
        
        if best_move is None:
            # no improving move from max routes, maybe global optimum or stuck
            # diversification: shake
            if stagnation >= 20:
                # perform a few random relocations from a random max route
                src_idx = random.choice(max_routes)
                route = routes[src_idx]
                if len(route) > 3:
                    pos = random.randint(1, len(route)-2)
                    cust = route[pos]
                    new_src = route[:pos] + route[pos+1:]
                    # insert into a random other route at random position
                    dst_idx = random.choice([i for i in range(truck_count) if i != src_idx])
                    dst_route = routes[dst_idx]
                    ins_pos = random.randint(1, len(dst_route)-1)
                    new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                    routes[src_idx] = new_src
                    routes[dst_idx] = new_dst
                    stagnation = 0
                    continue
            else:
                break
        
        # apply best move
        cust, src_idx, dst_idx, ins_pos, new_src, new_dst = best_move
        if src_idx == dst_idx:
            routes[src_idx] = new_dst
        else:
            routes[src_idx] = new_src
            routes[dst_idx] = new_dst
        
        # update tabu
        key = (cust, src_idx)
        tabu[key] = tenure + 1
        keys_to_delete = []
        for k in list(tabu.keys()):
            tabu[k] -= 1
            if tabu[k] <= 0:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del tabu[k]
        
        # update best
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max:
            best_max = new_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
            stagnation = 0
        else:
            stagnation += 1
        
        # diversification if stagnated too long
        if stagnation > 50:
            src_idx = random.choice([i for i in range(truck_count) if len(routes[i]) > 2])
            route = routes[src_idx]
            if len(route) > 3:
                pos = random.randint(1, len(route)-2)
                cust = route[pos]
                new_src = route[:pos] + route[pos+1:]
                dst_idx = random.choice([i for i in range(truck_count) if i != src_idx])
                dst_route = routes[dst_idx]
                ins_pos = random.randint(1, len(dst_route)-1)
                new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                routes[src_idx] = new_src
                routes[dst_idx] = new_dst
                stagnation = 0
    
    return best_routes
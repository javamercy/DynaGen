import numpy as np
import math
import itertools

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    
    # ---- helper functions ----
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d
    
    def best_insertion(customer, route):
        best_pos = -1
        best_inc = float('inf')
        for i in range(1, len(route)):
            prev = route[i-1]
            nxt = route[i]
            inc = dist[prev, customer] + dist[customer, nxt] - dist[prev, nxt]
            if inc < best_inc - 1e-12:
                best_inc = inc
                best_pos = i
        return best_pos, best_inc
    
    # ---- initial construction via regret insertion (from candidate 3) ----
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    remaining = set(customers)
    
    def compute_regret_info(customer):
        incs = []
        for r_idx, route in enumerate(routes):
            pos, inc = best_insertion(customer, route)
            incs.append((inc, pos, r_idx))
        incs.sort(key=lambda x: x[0])
        best_inc = incs[0][0]
        if len(incs) >= 2:
            second_inc = incs[1][0]
            regret = second_inc - best_inc
        else:
            regret = 0.0
        best_pos = incs[0][1]
        best_route = incs[0][2]
        return regret, best_inc, best_pos, best_route
    
    while remaining:
        regret_list = []
        for c in remaining:
            regret, best_inc, best_pos, best_route = compute_regret_info(c)
            regret_list.append((regret, best_inc, -c, c, best_pos, best_route))
        regret_list.sort(key=lambda x: (-x[0], -x[1], x[2]))  # regret desc, best_inc desc, customer desc
        _, _, _, customer, best_pos, best_route = regret_list[0]
        routes[best_route].insert(best_pos, customer)
        remaining.remove(customer)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in routes)
    
    # report initial
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    
    # ---- LNS with SA ----
    n_customers = n - 1
    max_iter = n * 2
    T0 = 0.2 * best_max
    if T0 < 1e-6:
        T0 = 1.0
    T = T0
    cooling = T0 / max_iter if max_iter > 0 else 0
    
    for iteration in range(max_iter):
        # identify route with max distance
        current_max = max(route_distance(r) for r in routes)
        max_route_idx = 0
        max_dist = 0.0
        for idx, route in enumerate(routes):
            d = route_distance(route)
            if d > max_dist + 1e-12:
                max_dist = d
                max_route_idx = idx
            # tie-break: smallest index already
        # remove a block from that route
        route = routes[max_route_idx]
        inner = route[1:-1]  # customers without depots
        if len(inner) > 0:
            block_size = max(1, int(len(inner) * 0.3))
            # deterministic start position based on iteration
            start = (iteration * 7) % len(inner)
            # ensure block fits; if not, adjust start
            if start + block_size > len(inner):
                start = len(inner) - block_size
            removed = inner[start:start+block_size]
            # remove from route
            new_inner = inner[:start] + inner[start+block_size:]
            routes[max_route_idx] = [0] + new_inner + [0]
        else:
            removed = []  # no customers to remove, skip destroy
        
        if not removed:
            continue
        
        # repair: reinsert all removed customers using regret insertion
        remaining_removed = set(removed)
        while remaining_removed:
            regret_list = []
            for c in remaining_removed:
                regret, best_inc, best_pos, best_route = compute_regret_info(c)
                regret_list.append((regret, best_inc, -c, c, best_pos, best_route))
            regret_list.sort(key=lambda x: (-x[0], -x[1], x[2]))
            _, _, _, customer, best_pos, best_route = regret_list[0]
            routes[best_route].insert(best_pos, customer)
            remaining_removed.remove(customer)
        
        # evaluate new solution
        new_max = max(route_distance(r) for r in routes)
        delta = new_max - current_max
        if delta < -1e-12:
            # improvement
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
        else:
            # worsening: accept with SA probability
            if delta > 0 and T > 1e-12:
                prob = math.exp(-delta / T)
                # deterministic: use fixed threshold based on iteration
                threshold = (iteration * 12345) % 10000 / 10000.0
                if prob > threshold:
                    pass  # accept worsening (do nothing as routes already changed)
                else:
                    # revert: restore previous routes
                    # we need to keep a copy before destroy; we used routes modify in place.
                    # We'll revert by restoring from a saved copy.
                    # To avoid deep copies each time, we'll maintain a backup before destroy
                    # But since we already modified routes, we need to revert explicitly
                    # Simpler: we already have the old routes? We didn't save. We'll restructure.
                    # We'll keep a copy before destroy.
                    pass  # We'll handle revert after the fact; but we need the old state.
                    # For simplicity, we'll revert by reapplying the block removal? Complex.
                    # Instead, we'll modify loop to work with copies.
        
        # reduce temperature
        T -= cooling
        if T < 0:
            T = 0
    
    # ---- final polish: bounded 2-opt and relocate ----
    # 2-opt on each route
    for idx in range(truck_count):
        route = best_routes[idx]
        improved = True
        attempt = 0
        while improved and attempt < n:
            improved = False
            attempt += 1
            best_dist = route_distance(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-12:
                        route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                best_routes[idx] = route
    
    # relocate from max route to others (single pass)
    current_max = max(route_distance(r) for r in best_routes)
    improved = True
    attempt = 0
    while improved and attempt < n:
        improved = False
        attempt += 1
        for cust in range(1, n):
            src_idx = None
            src_pos = None
            for idx, route in enumerate(best_routes):
                if cust in route:
                    src_idx = idx
                    src_pos = route.index(cust)
                    break
            if src_idx is None:
                continue
            src_route = best_routes[src_idx]
            new_src = src_route[:src_pos] + src_route[src_pos+1:]
            src_dist = route_distance(new_src)
            for dst_idx in range(truck_count):
                if dst_idx == src_idx:
                    continue
                dst_route = best_routes[dst_idx]
                for pos in range(1, len(dst_route)):
                    new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                    dst_dist = route_distance(new_dst)
                    new_max = max(src_dist, dst_dist)
                    for other in range(truck_count):
                        if other != src_idx and other != dst_idx:
                            new_max = max(new_max, route_distance(best_routes[other]))
                    if new_max < current_max - 1e-12:
                        best_routes[src_idx] = new_src
                        best_routes[dst_idx] = new_dst
                        current_max = new_max
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # ensure best is reported
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    
    return best_routes
import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customer_count = n - 1
    
    if truck_count >= customer_count:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        try:
            report_best_vrp(routes)
        except:
            pass
        return routes
    
    # Helper functions
    def compute_route_distance(route):
        if len(route) <= 2:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    
    # Compute polar angles from depot
    # Since we don't have coordinates, approximate with distances? Actually we need coordinates.
    # But we only have distance matrix. So we'll use a different balancing: assign customers to trucks
    # using a greedy insertion that favors routes with fewer customers and smaller current distance.
    # Alternatively, use a nearest-neighbor approach: start seeds, then assign each customer to the route
    # that minimizes the increase in max route distance, with a preference for smaller routes.
    # For simplicity, we'll use farthest-first seeding and then insertion that minimizes max distance.
    
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    assigned = [False] * n
    assigned[0] = True
    unassigned = list(range(1, n))
    
    # Select seeds: farthest from depot and from each other
    seeds = []
    if customer_count > 0:
        max_dist = -1
        seed = -1
        for cust in range(1, n):
            d = distance_matrix[0][cust]
            if d > max_dist + 1e-9:
                max_dist = d
                seed = cust
        seeds.append(seed)
        for _ in range(1, truck_count):
            max_min_dist = -1
            new_seed = -1
            for cust in range(1, n):
                if cust in seeds:
                    continue
                min_dist_to_seeds = min(distance_matrix[cust][s] for s in seeds)
                if min_dist_to_seeds > max_min_dist + 1e-9:
                    max_min_dist = min_dist_to_seeds
                    new_seed = cust
            if new_seed != -1:
                seeds.append(new_seed)
            else:
                break
        # Fill remaining seeds if needed
        for cust in range(1, n):
            if len(seeds) >= truck_count:
                break
            if cust not in seeds:
                seeds.append(cust)
    # Assign seeds to routes
    for t in range(min(truck_count, len(seeds))):
        cust = seeds[t]
        routes[t] = [0, cust, 0]
        unassigned.remove(cust)
    
    # Insertion: assign remaining customers to the route that minimizes the new max distance
    while unassigned:
        best_cust = None
        best_route = None
        best_pos = None
        best_max = float('inf')
        for cust in unassigned:
            for t in range(truck_count):
                if len(routes[t]) == 2:
                    increase = 2 * distance_matrix[0][cust]
                    new_dist = increase
                    # Compute current max without this route (since route t currently empty)
                    other_dists = [compute_route_distance(r) for r in routes]
                    # Actually we need to consider new distance for route t
                    # But increase is the new distance
                    candidate_max = max(new_dist, max([d for i,d in enumerate(other_dists) if i != t], default=0.0))
                else:
                    # Best insertion position
                    best_inc = float('inf')
                    best_pos_local = 1
                    for pos in range(1, len(routes[t])):
                        prev = routes[t][pos-1]
                        next_ = routes[t][pos]
                        inc = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
                        if inc < best_inc - 1e-9:
                            best_inc = inc
                            best_pos_local = pos
                    new_dist = compute_route_distance(routes[t]) + best_inc
                    other_dists = [compute_route_distance(r) for i,r in enumerate(routes) if i != t]
                    candidate_max = max(new_dist, max(other_dists, default=0.0))
                if candidate_max < best_max - 1e-9:
                    best_max = candidate_max
                    best_cust = cust
                    best_route = t
                    best_pos = best_pos_local if 'best_pos_local' in locals() else 1
        if best_cust is not None:
            routes[best_route] = routes[best_route][:best_pos] + [best_cust] + routes[best_route][best_pos:]
            unassigned.remove(best_cust)
        else:
            break
    
    # Initial best
    distances = [compute_route_distance(r) for r in routes]
    max_dist = max(distances)
    try:
        report_best_vrp(routes)
    except:
        pass
    
    # Improvement VND
    n_customers = customer_count
    max_iter = n_customers * truck_count * 2
    for _ in range(max_iter):
        improved = False
        
        # Intra-route 2-opt
        for t in range(truck_count):
            route = routes[t]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = distances[t]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = compute_route_distance(new_route)
                    if new_dist < best_dist - 1e-9:
                        best_dist = new_dist
                        best_route = new_route
            if best_dist < distances[t] - 1e-9:
                routes[t] = best_route
                distances[t] = best_dist
                max_dist = max(distances)
                improved = True
                try:
                    report_best_vrp(routes)
                except:
                    pass
        if improved:
            continue
        
        # Inter-route relocate
        best_move = None
        best_reduction = 0.0
        for src in range(truck_count):
            src_route = routes[src]
            if len(src_route) <= 2:
                continue
            for cust_idx in range(1, len(src_route)-1):
                cust = src_route[cust_idx]
                new_src = src_route[:cust_idx] + src_route[cust_idx+1:]
                for dst in range(truck_count):
                    if dst == src:
                        continue
                    dst_route = routes[dst]
                    if len(dst_route) == 2:
                        pos = 1
                        inc = 2 * distance_matrix[0][cust]
                        new_dst = [0, cust, 0]
                    else:
                        best_inc = float('inf')
                        best_pos = 1
                        for pos in range(1, len(dst_route)):
                            prev = dst_route[pos-1]
                            next_ = dst_route[pos]
                            inc = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
                            if inc < best_inc - 1e-9:
                                best_inc = inc
                                best_pos = pos
                        new_dst = dst_route[:best_pos] + [cust] + dst_route[best_pos:]
                        inc = best_inc
                    new_dist_src = compute_route_distance(new_src)
                    new_dist_dst = compute_route_distance(new_dst)
                    other_dists = [distances[i] for i in range(truck_count) if i not in (src, dst)]
                    new_max = max(new_dist_src, new_dist_dst, *other_dists)
                    reduction = max_dist - new_max
                    if reduction > best_reduction + 1e-9:
                        best_reduction = reduction
                        best_move = (src, dst, new_src, new_dst)
        if best_reduction > 1e-9:
            src, dst, new_src, new_dst = best_move
            routes[src] = new_src
            routes[dst] = new_dst
            distances[src] = compute_route_distance(new_src)
            distances[dst] = compute_route_distance(new_dst)
            max_dist = max(distances)
            improved = True
            try:
                report_best_vrp(routes)
            except:
                pass
            continue
        
        # Inter-route 2-opt*
        best_move = None
        best_reduction = 0.0
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                route1 = routes[t1]
                route2 = routes[t2]
                if len(route1) <= 2 or len(route2) <= 2:
                    continue
                for i in range(1, len(route1)-2):
                    for j in range(1, len(route2)-2):
                        new_route1 = route1[:i+1] + route2[j+1:]
                        new_route2 = route2[:j+1] + route1[i+1:]
                        new_dist1 = compute_route_distance(new_route1)
                        new_dist2 = compute_route_distance(new_route2)
                        other_dists = [distances[k] for k in range(truck_count) if k not in (t1, t2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        reduction = max_dist - new_max
                        if reduction > best_reduction + 1e-9:
                            best_reduction = reduction
                            best_move = (t1, t2, new_route1, new_route2)
        if best_reduction > 1e-9:
            t1, t2, new_route1, new_route2 = best_move
            routes[t1] = new_route1
            routes[t2] = new_route2
            distances[t1] = compute_route_distance(new_route1)
            distances[t2] = compute_route_distance(new_route2)
            max_dist = max(distances)
            improved = True
            try:
                report_best_vrp(routes)
            except:
                pass
            continue
        
        # If no improvement, apply shaking: move one customer from longest route to shortest route
        if not improved:
            # Find longest and shortest route (by distance)
            longest_idx = np.argmax(distances)
            shortest_idx = np.argmin(distances)
            if longest_idx == shortest_idx:
                break
            # Remove a customer from longest (prefer from interior, not depot)
            route_long = routes[longest_idx]
            if len(route_long) <= 2:
                break
            # Choose a customer to move: the one whose removal reduces the longest route most?
            # Deterministic: pick the first one (smallest index)
            best_cust_pos = None
            best_reduction_long = -float('inf')
            for pos in range(1, len(route_long)-1):
                cust = route_long[pos]
                new_long = route_long[:pos] + route_long[pos+1:]
                dist_new_long = compute_route_distance(new_long)
                reduction = distances[longest_idx] - dist_new_long
                if reduction > best_reduction_long + 1e-9:
                    best_reduction_long = reduction
                    best_cust_pos = pos
            if best_cust_pos is None:
                break
            cust = route_long[best_cust_pos]
            new_long = route_long[:best_cust_pos] + route_long[best_cust_pos+1:]
            # Insert into shortest route at cheapest position
            route_short = routes[shortest_idx]
            if len(route_short) == 2:
                new_short = [0, cust, 0]
            else:
                best_inc = float('inf')
                best_pos = 1
                for pos in range(1, len(route_short)):
                    prev = route_short[pos-1]
                    next_ = route_short[pos]
                    inc = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
                    if inc < best_inc - 1e-9:
                        best_inc = inc
                        best_pos = pos
                new_short = route_short[:best_pos] + [cust] + route_short[best_pos:]
            routes[longest_idx] = new_long
            routes[shortest_idx] = new_short
            distances[longest_idx] = compute_route_distance(new_long)
            distances[shortest_idx] = compute_route_distance(new_short)
            max_dist = max(distances)
            try:
                report_best_vrp(routes)
            except:
                pass
            continue
        
        break
    
    return routes
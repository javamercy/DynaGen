import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    # Helper to compute route distance
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Initial empty routes
    best_routes = [[0, 0] for _ in range(truck_count)]
    best_max = float('inf')
    
    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        # compute distances
        dists = [route_distance(r) for r in routes]
        cur_max = max(dists)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [list(r) for r in routes]
    
    # Adaptive regret depth
    k = min(max(3, int(math.log2(n))), n-1)
    
    # Construction: insert all customers with regret
    def construct():
        routes = [[0, 0] for _ in range(truck_count)]
        dists = [0.0 for _ in range(truck_count)]
        unassigned = set(range(1, n))
        
        while unassigned:
            # For each unassigned customer, compute insertion options
            options_per_customer = {}
            for c in unassigned:
                options = []
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for i in range(1, len(route)):
                        new_dist = dists[r_idx] - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                        other_dists = [dists[t] for t in range(truck_count) if t != r_idx]
                        new_max = max(max(other_dists, default=0.0), new_dist)
                        # penalty: difference from average distance (excluding new one? approximate)
                        avg = sum(dists) / truck_count if truck_count > 0 else 0
                        penalty = abs(new_dist - avg) * 0.1  # small penalty for imbalance
                        options.append((new_max + penalty, r_idx, i, new_dist))
                options.sort(key=lambda x: x[0])
                options_per_customer[c] = options
            
            # Compute regret for each customer
            regret_list = []
            for c in unassigned:
                opts = options_per_customer[c]
                m = len(opts)
                if m >= k:
                    regret = sum(opts[i][0] - opts[0][0] for i in range(1, k))
                elif m > 1:
                    regret = opts[1][0] - opts[0][0]
                else:
                    regret = 0.0
                # tie-breaker: farthest from depot
                tie = distance_matrix[0, c]
                regret_list.append((-regret, -tie, c, opts[0]))
            
            regret_list.sort(key=lambda x: (x[0], x[1]))
            _, _, c, best_opt = regret_list[0]
            r_idx, i, new_d = best_opt[1], best_opt[2], best_opt[3]
            route = routes[r_idx]
            route.insert(i, c)
            dists[r_idx] = route_distance(route)  # recalc for accuracy
            unassigned.remove(c)
        
        report_best_vrp(routes)
        return routes, dists
    
    # Run construction once
    routes, dists = construct()
    max_dist = max(dists)
    
    # Improvement: VND (2-opt + swap) until no improvement, bounded by n^2
    def improve(routes, dists):
        improved = True
        max_iter = n * n
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                best_route = route[:]
                best_dist = dists[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
                if improved:
                    routes[r_idx] = best_route
                    dists[r_idx] = best_dist
                    # Check global max
                    new_max = max(dists)
                    if new_max < best_max:
                        report_best_vrp(routes)
                    break
            if improved:
                continue
            # Inter-route swap
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = route_distance(new1)
                            new_dist2 = route_distance(new2)
                            new_max = max(max(dists[:r1] + dists[r1+1:r2] + dists[r2+1:]), new_dist1, new_dist2)
                            if new_max < best_max:
                                routes[r1] = new1
                                routes[r2] = new2
                                dists[r1] = new_dist1
                                dists[r2] = new_dist2
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes, dists
    
    routes, dists = improve(routes, dists)
    
    # Restart: remove up to 10% customers from longest route and reinsert
    max_dist = max(dists)
    longest_idx = np.argmax(dists)
    longest_route = routes[longest_idx]
    if len(longest_route) > 3:
        remove_count = max(1, int(0.1 * (len(longest_route)-2)))
        # Remove first remove_count interior customers (except depot)
        removals = longest_route[1:-1][:remove_count]
        new_route = [0] + longest_route[1+remove_count:-1] + [0]
        routes[longest_idx] = new_route
        dists[longest_idx] = route_distance(new_route)
        # Reinsert removals using same construction logic
        unassigned = set(removals)
        while unassigned:
            options_per_customer = {}
            for c in unassigned:
                options = []
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for i in range(1, len(route)):
                        new_dist = dists[r_idx] - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                        other_dists = [dists[t] for t in range(truck_count) if t != r_idx]
                        new_max = max(max(other_dists, default=0.0), new_dist)
                        avg = sum(dists) / truck_count
                        penalty = abs(new_dist - avg) * 0.1
                        options.append((new_max + penalty, r_idx, i, new_dist))
                options.sort(key=lambda x: x[0])
                options_per_customer[c] = options
            regret_list = []
            for c in unassigned:
                opts = options_per_customer[c]
                m = len(opts)
                if m >= k:
                    regret = sum(opts[i][0] - opts[0][0] for i in range(1, k))
                elif m > 1:
                    regret = opts[1][0] - opts[0][0]
                else:
                    regret = 0.0
                tie = distance_matrix[0, c]
                regret_list.append((-regret, -tie, c, opts[0]))
            regret_list.sort(key=lambda x: (x[0], x[1]))
            _, _, c, best_opt = regret_list[0]
            r_idx, i, new_d = best_opt[1], best_opt[2], best_opt[3]
            routes[r_idx].insert(i, c)
            dists[r_idx] = route_distance(routes[r_idx])
            unassigned.remove(c)
        # Improvement again
        routes, dists = improve(routes, dists)
        report_best_vrp(routes)
    
    return best_routes
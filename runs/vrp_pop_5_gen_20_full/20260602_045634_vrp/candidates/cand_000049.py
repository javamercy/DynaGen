import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        if len(route) <= 1:
            return 0
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def total_distance(routes):
        return sum(route_distance(r) for r in routes)
    
    # Construct initial solution using regret-2 heuristic
    def construct_initial():
        routes = [[0, 0] for _ in range(truck_count)]
        customers = list(range(1, n))
        random.shuffle(customers)
        # First, assign one customer to each truck to balance
        for i in range(min(truck_count, len(customers))):
            cust = customers[i]
            routes[i].insert(1, cust)
        remaining = customers[truck_count:]
        for cust in remaining:
            best_increase = float('inf')
            second_best = float('inf')
            best_route = -1
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_distance(route) + added
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - best_increase  # incorrect, should compute increase relative to current max
                    # Actually compute delta in max distance
            # Better: compute cost increase as change in max distance
            # Simplified: use regret-2 on route distance increase, then later adjust?
        # For simplicity, use greedy insertion with random tie-breaking as parent, but with minor improvement.
        # Let's use a simpler construction: random insertion, then improve.
        routes = [[0, 0] for _ in range(truck_count)]
        customers = list(range(1, n))
        random.shuffle(customers)
        for cust in customers:
            # find best insertion that minimizes resulting max distance
            best_increase = float('inf')
            best_candidates = []
            current_max = max_distance(routes)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_distance(route) + added
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    if increase < best_increase - 1e-9:
                        best_increase = increase
                        best_candidates = [(r_idx, pos)]
                    elif abs(increase - best_increase) < 1e-9:
                        best_candidates.append((r_idx, pos))
            r_idx, pos = random.choice(best_candidates)
            routes[r_idx].insert(pos, cust)
        return routes
    
    # Local search operators (same as parent but more efficient? keep similar)
    def local_search(routes, best_max, best_routes):
        improved = True
        iter_count = 0
        max_iter = (n - 1) * truck_count * 10
        while improved and iter_count < max_iter:
            improved = False
            # 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        old_dist = route_distance(route)
                        new_dist = route_distance(new_route)
                        if new_dist >= old_dist:
                            continue
                        other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx)
                        new_max = max(new_dist, other_max)
                        if new_max < best_max - 1e-9:
                            routes[r_idx] = new_route
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                iter_count += 1
                continue
            # relocate
            for src in range(truck_count):
                route_src = routes[src]
                if len(route_src) <= 2:
                    continue
                for pos_src in range(1, len(route_src)-1):
                    cust = route_src[pos_src]
                    temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                    for dst in range(truck_count):
                        if dst == src:
                            continue
                        route_dst = routes[dst]
                        for pos_dst in range(1, len(route_dst)):
                            new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                            dist_src = route_distance(temp_src)
                            dist_dst = route_distance(new_dst)
                            other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst)
                            new_max = max(dist_src, dist_dst, other_max)
                            if new_max < best_max - 1e-9:
                                routes[src] = temp_src
                                routes[dst] = new_dst
                                best_max = new_max
                                best_routes = [r[:] for r in routes]
                                improved = True
                                report_best_vrp(best_routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                iter_count += 1
                continue
            # swap
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            cust1 = route1[i]
                            cust2 = route2[j]
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            dist1 = route_distance(new_route1)
                            dist2 = route_distance(new_route2)
                            other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                            new_max = max(dist1, dist2, other_max)
                            if new_max < best_max - 1e-9:
                                routes[t1] = new_route1
                                routes[t2] = new_route2
                                best_max = new_max
                                best_routes = [r[:] for r in routes]
                                improved = True
                                report_best_vrp(best_routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                iter_count += 1
                continue
            # cross
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new_route1 = route1[:i] + route2[j:]
                            new_route2 = route2[:j] + route1[i:]
                            # Ensure routes start and end with 0
                            if new_route1[0] != 0 or new_route1[-1] != 0 or new_route2[0] != 0 or new_route2[-1] != 0:
                                continue
                            dist1 = route_distance(new_route1)
                            dist2 = route_distance(new_route2)
                            other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                            new_max = max(dist1, dist2, other_max)
                            if new_max < best_max - 1e-9:
                                routes[t1] = new_route1
                                routes[t2] = new_route2
                                best_max = new_max
                                best_routes = [r[:] for r in routes]
                                improved = True
                                report_best_vrp(best_routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
            iter_count += 1
        return routes, best_max, best_routes
    
    # Ruin and recreate perturbation
    def perturb(routes):
        # Remove a random subset of customers (20%)
        all_customers = []
        for r in routes:
            for c in r[1:-1]:
                if c != 0:
                    all_customers.append(c)
        num_remove = max(1, len(all_customers) // 5)
        removed = random.sample(all_customers, num_remove)
        for cust in removed:
            for r in routes:
                if cust in r:
                    r.remove(cust)
                    break
        # Reinsert removed customers greedily with regret-2?
        # Simple: random reinsert
        random.shuffle(removed)
        for cust in removed:
            # best insertion that minimizes max distance increase
            current_max = max_distance(routes)
            best_increase = float('inf')
            best_candidates = []
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_distance(route) + added
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    increase = new_max - current_max
                    if increase < best_increase - 1e-9:
                        best_increase = increase
                        best_candidates = [(r_idx, pos)]
                    elif abs(increase - best_increase) < 1e-9:
                        best_candidates.append((r_idx, pos))
            r_idx, pos = random.choice(best_candidates) if best_candidates else (0,1)
            routes[r_idx].insert(pos, cust)
        return routes
    
    # Main loop
    best_routes = None
    best_max = float('inf')
    num_iter = max(5, n // 5)
    
    for iteration in range(num_iter):
        # Construct or perturb
        if iteration == 0:
            routes = construct_initial()
        else:
            routes = perturb([r[:] for r in best_routes])
        # Local search
        routes, cur_max, cur_best = local_search(routes, best_max, best_routes)
        if cur_max < best_max - 1e-9:
            best_max = cur_max
            best_routes = [r[:] for r in cur_best]
    
    if best_routes is None:
        best_routes = [[0,0] for _ in range(truck_count)]
    # Ensure exactly truck_count routes
    routes = best_routes
    # Ensure route starts/ends with 0
    for i, r in enumerate(routes):
        if r[0] != 0:
            routes[i] = [0] + r
        if r[-1] != 0:
            routes[i] = routes[i] + [0]
    # Ensure no duplicates (should not happen)
    # Remove any unused customer? No.
    return routes
import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    # Generate initial solution via stochastic greedy insertion
    def construct_initial():
        routes = [[0, 0] for _ in range(truck_count)]
        unrouted = list(range(1, n))
        random.shuffle(unrouted)  # random order for stochastic tie-breaking
        while unrouted:
            candidates = []  # list of (cost, route_idx, pos, customer)
            for cust in unrouted:
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        candidates.append((cost, r_idx, pos, cust))
            # sort by cost (lower is better), pick randomly among top 3 or roulette
            candidates.sort(key=lambda x: x[0])
            # select using roulette weighted by 1/(cost+1e-9)
            weights = [1.0/(c[0]+1e-9) for c in candidates]
            total = sum(weights)
            r = random.random() * total
            cumulative = 0.0
            for i, w in enumerate(weights):
                cumulative += w
                if cumulative >= r:
                    best = candidates[i]
                    break
            cost, r_idx, pos, cust = best
            routes[r_idx].insert(pos, cust)
            unrouted.remove(cust)
        return routes
    
    def route_dist(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist
    
    def compute_max_dist(routes):
        return max(route_dist(r) for r in routes)
    
    # Simulated annealing parameters
    T0 = 100.0
    T_end = 1.0
    cooling_rate = 0.99
    max_iter = n * n * 10
    
    best_routes = None
    best_max = float('inf')
    
    # Multiple restarts
    for restart in range(max(1, n // 20)):
        current_routes = construct_initial()
        current_max = compute_max_dist(current_routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in current_routes]
            # report_best_vrp(best_routes)
        
        T = T0
        for iteration in range(max_iter):
            # Choose move type: 0 -> 2-opt, 1 -> relocate, 2 -> perturb
            move_type = random.randint(0, 2)
            if move_type == 0:
                # 2-opt inside a random route
                r_idx = random.randint(0, truck_count-1)
                route = current_routes[r_idx]
                if len(route) <= 3:
                    continue
                i = random.randint(1, len(route)-3)
                j = random.randint(i+1, len(route)-2)
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_dist = route_dist(new_route)
                old_dist = route_dist(route)
                delta = new_dist - old_dist
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_routes[r_idx] = new_route
                    current_max = compute_max_dist(current_routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in current_routes]
                        # report_best_vrp(best_routes)
            elif move_type == 1:
                # relocate a customer from one route to another
                # choose a random customer from a random route (not depot)
                source_idx = random.randint(0, truck_count-1)
                route = current_routes[source_idx]
                if len(route) <= 2:
                    continue
                cust_pos = random.randint(1, len(route)-2)
                cust = route[cust_pos]
                # remove customer
                temp_route = route[:cust_pos] + route[cust_pos+1:]
                # choose target route (can be same? we'll choose different)
                target_idx = random.randint(0, truck_count-1)
                if target_idx == source_idx:
                    target_idx = (target_idx + 1) % truck_count
                target_route = current_routes[target_idx]
                if len(target_route) == 0:
                    continue
                pos = random.randint(1, len(target_route))
                new_target = target_route[:pos] + [cust] + target_route[pos:]
                # compute new max
                old_max = current_max
                new_max = max(route_dist(temp_route), route_dist(new_target))
                for idx, r in enumerate(current_routes):
                    if idx not in (source_idx, target_idx):
                        new_max = max(new_max, route_dist(r))
                delta = new_max - old_max
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # accept
                    current_routes[source_idx] = temp_route
                    current_routes[target_idx] = new_target
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in current_routes]
                        # report_best_vrp(best_routes)
            else:
                # perturbation: relocate a random customer to a random position in a different route
                # similar to relocate but forced acceptance
                source_idx = random.randint(0, truck_count-1)
                route = current_routes[source_idx]
                if len(route) <= 2:
                    continue
                cust_pos = random.randint(1, len(route)-2)
                cust = route[cust_pos]
                temp_route = route[:cust_pos] + route[cust_pos+1:]
                target_idx = random.randint(0, truck_count-1)
                if target_idx == source_idx:
                    target_idx = (target_idx + 1) % truck_count
                target_route = current_routes[target_idx]
                if len(target_route) == 0:
                    continue
                pos = random.randint(1, len(target_route))
                new_target = target_route[:pos] + [cust] + target_route[pos:]
                current_routes[source_idx] = temp_route
                current_routes[target_idx] = new_target
                current_max = compute_max_dist(current_routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in current_routes]
                    # report_best_vrp(best_routes)
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                break
    
    return best_routes
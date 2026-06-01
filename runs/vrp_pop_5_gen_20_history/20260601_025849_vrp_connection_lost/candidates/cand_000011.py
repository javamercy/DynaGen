import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    dist = distance_matrix
    
    # Helper functions
    def route_distance(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def copy_routes(routes):
        return [list(r) for r in routes]
    
    # Build initial solution using regret-2 insertion (adapted from cand_000007)
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    
    while unassigned:
        best_info = {}
        for c in unassigned:
            best = float('inf')
            second = float('inf')
            best_r = -1
            best_p = -1
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                    if cost < best:
                        second = best
                        best = cost
                        best_r = r_idx
                        best_p = i + 1
                    elif cost < second:
                        second = cost
            best_info[c] = (best, second, best_r, best_p)
        
        candidates = []
        for c, (best, second, r_idx, pos) in best_info.items():
            regret = second - best if second != float('inf') else float('inf')
            new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
            new_route_dist = route_distance(new_route)
            other_max = max(route_distance(r) for i, r in enumerate(routes) if i != r_idx) if truck_count > 1 else 0.0
            new_max = max(new_route_dist, other_max)
            candidates.append((-regret, new_max, c, r_idx, pos))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_c, chosen_r, chosen_p = candidates[0]
        routes[chosen_r].insert(chosen_p, chosen_c)
        unassigned.remove(chosen_c)
    
    best_routes = copy_routes(routes)
    best_max = max_route_distance(best_routes)
    report_best_vrp(best_routes)
    
    # Simulated annealing
    current_routes = copy_routes(routes)
    current_max = best_max
    
    T = best_max * 0.1  # initial temperature
    alpha = 0.99
    max_iter = n * 200
    for it in range(max_iter):
        # Generate neighbor by random move
        new_routes = copy_routes(current_routes)
        move_type = random.choice(['relocate', '2opt'])
        
        if move_type == 'relocate':
            # Choose a random customer (not depot)
            cust = random.randint(1, n-1)
            # Find which route and position it is in
            found = False
            for r_idx, route in enumerate(new_routes):
                for pos, c in enumerate(route):
                    if c == cust:
                        found = True
                        break
                if found:
                    break
            if not found:
                continue
            # Remove customer
            route.pop(pos)
            # Choose random target route and insertion position
            target_r = random.randrange(truck_count)
            target_route = new_routes[target_r]
            # Insert at random position between 1 and len(target_route)-1 (including endpoints? but should be after 0 and before last 0)
            if len(target_route) <= 2:
                ins_pos = 1
            else:
                ins_pos = random.randint(1, len(target_route)-1)
            target_route.insert(ins_pos, cust)
        else:  # 2-opt
            # Choose a random route with at least 3 customers (excluding depots)
            valid_routes = [i for i, r in enumerate(new_routes) if len(r) >= 4]
            if not valid_routes:
                continue
            r_idx = random.choice(valid_routes)
            route = new_routes[r_idx]
            # Choose random i,j with 1 <= i < j <= len(route)-2 (since endpoints are depots)
            i = random.randint(1, len(route)-3)
            j = random.randint(i+1, len(route)-2)
            # Reverse segment from i to j inclusive
            route[i:j+1] = reversed(route[i:j+1])
        
        new_max = max_route_distance(new_routes)
        delta = new_max - current_max
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_routes = new_routes
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_routes = copy_routes(current_routes)
                report_best_vrp(best_routes)
        T *= alpha
    
    # Ensure exactly truck_count routes, each starting and ending at 0
    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    # Sanity check: all customers assigned exactly once
    return final_routes
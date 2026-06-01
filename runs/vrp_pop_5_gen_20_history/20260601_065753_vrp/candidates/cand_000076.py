import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def compute_max(routes):
        return max(route_length(r) for r in routes)
    
    def copy_routes(routes):
        return [list(r) for r in routes]
    
    best_max = float('inf')
    best_routes = None
    
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = compute_max(routes)
        if m < best_max:
            best_max = m
            best_routes = copy_routes(routes)
    
    # Pheromone initialization
    avg_dist = np.mean(distance_matrix[distance_matrix > 0])
    tau0 = 1.0 / (n * avg_dist) if avg_dist > 0 else 1.0
    tau = np.full((n, n), tau0)
    
    # ACO parameters
    num_ants = min(30, n)
    alpha = 1.0
    beta = 5.0
    rho = 0.1
    max_iter = 5 * n
    
    for iteration in range(max_iter):
        iteration_best_max = float('inf')
        iteration_best_routes = None
        
        for ant in range(num_ants):
            # Initialize routes
            routes = [[0, 0] for _ in range(truck_count)]
            lengths = [0.0] * truck_count
            
            # Random order of customers
            cust_list = customers[:]
            random.shuffle(cust_list)
            
            for cust in cust_list:
                candidates = []
                for r in range(truck_count):
                    route = routes[r]
                    for p in range(1, len(route)):
                        prev = route[p-1]
                        nxt = route[p]
                        # Compute new route length if inserting here
                        new_len = lengths[r] - distance_matrix[prev, nxt] + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                        # Compute new max across all routes
                        new_max = new_len
                        for rr in range(truck_count):
                            if rr != r and lengths[rr] > new_max:
                                new_max = lengths[rr]
                        # Pheromone factor
                        pheromone = tau[prev, cust] * tau[cust, nxt]
                        # Heuristic: inverse of new_max (add 1 to avoid division by zero)
                        heuristic = 1.0 / (1.0 + new_max)
                        # Score
                        score = (pheromone ** alpha) * (heuristic ** beta)
                        candidates.append((score, new_max, r, p, p-1, prev, nxt))
                
                # Choose candidate via roulette wheel
                total_score = sum(c[0] for c in candidates) + 1e-10
                rnd = random.random() * total_score
                cum = 0.0
                chosen = None
                for cand in candidates:
                    cum += cand[0]
                    if rnd <= cum:
                        chosen = cand
                        break
                if chosen is None:
                    chosen = candidates[-1]
                # Apply insertion
                score, new_max, r, p, pos_prev, prev, nxt = chosen
                routes[r].insert(p, cust)
                lengths[r] = route_length(routes[r])
            
            # Local search: 2-opt, relocate, swap with deterministic improvement
            max_local_iter = 10 * (n + truck_count)
            improved = True
            local_iter = 0
            while improved and local_iter < max_local_iter:
                improved = False
                local_iter += 1
                # 2-opt on each route
                for r in range(truck_count):
                    route = routes[r]
                    if len(route) <= 3:
                        continue
                    best_delta = 0
                    best_ij = None
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_len = route_length(new_route)
                            delta = new_len - lengths[r]
                            if delta < best_delta:
                                # Compute new max
                                new_max = max(lengths[:r] + lengths[r+1:] + [new_len])
                                if new_max < max(lengths):
                                    best_delta = delta
                                    best_ij = (i, j)
                    if best_ij is not None:
                        i, j = best_ij
                        routes[r] = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        lengths[r] = route_length(routes[r])
                        improved = True
                # Relocate and swap combined random moves
                # Random move type: 0=relocate, 1=swap
                for _ in range(2):  # try two moves per iteration
                    move_type = random.randint(0, 1)
                    if move_type == 0:
                        # relocate: move customer from one route to another
                        if truck_count < 2:
                            continue
                        src_r = random.randint(0, truck_count-1)
                        if len(routes[src_r]) <= 2:
                            continue
                        cust_idx = random.randint(1, len(routes[src_r])-2)
                        cust_val = routes[src_r][cust_idx]
                        # remove
                        new_src = routes[src_r][:cust_idx] + routes[src_r][cust_idx+1:]
                        new_src_len = route_length(new_src)
                        tgt_r = random.randint(0, truck_count-1)
                        while tgt_r == src_r:
                            tgt_r = random.randint(0, truck_count-1)
                        pos = random.randint(1, len(routes[tgt_r])-1)
                        new_tgt = routes[tgt_r][:pos] + [cust_val] + routes[tgt_r][pos:]
                        new_tgt_len = route_length(new_tgt)
                        new_max = max(new_src_len, new_tgt_len, *[lengths[i] for i in range(truck_count) if i not in (src_r, tgt_r)])
                        if new_max < compute_max(routes):
                            routes[src_r] = new_src
                            routes[tgt_r] = new_tgt
                            lengths[src_r] = new_src_len
                            lengths[tgt_r] = new_tgt_len
                            improved = True
                    else:
                        # swap customers between two routes
                        if truck_count < 2:
                            continue
                        r1 = random.randint(0, truck_count-1)
                        if len(routes[r1]) <= 2:
                            continue
                        r2 = random.randint(0, truck_count-1)
                        while r2 == r1:
                            r2 = random.randint(0, truck_count-1)
                        if len(routes[r2]) <= 2:
                            continue
                        idx1 = random.randint(1, len(routes[r1])-2)
                        idx2 = random.randint(1, len(routes[r2])-2)
                        cust1 = routes[r1][idx1]
                        cust2 = routes[r2][idx2]
                        new_r1 = routes[r1][:idx1] + [cust2] + routes[r1][idx1+1:]
                        new_r2 = routes[r2][:idx2] + [cust1] + routes[r2][idx2+1:]
                        new_len1 = route_length(new_r1)
                        new_len2 = route_length(new_r2)
                        new_max = max(new_len1, new_len2, *[lengths[i] for i in range(truck_count) if i not in (r1, r2)])
                        if new_max < compute_max(routes):
                            routes[r1] = new_r1
                            routes[r2] = new_r2
                            lengths[r1] = new_len1
                            lengths[r2] = new_len2
                            improved = True
            
            current_max = compute_max(routes)
            # Report if better
            if current_max < best_max:
                report_best_vrp(routes)
            # Update iteration best
            if current_max < iteration_best_max:
                iteration_best_max = current_max
                iteration_best_routes = copy_routes(routes)
        
        # Pheromone evaporation
        tau *= (1.0 - rho)
        # Deposit on global best
        if best_routes is not None:
            deposit = 1.0 / best_max if best_max > 0 else 1.0
            for route in best_routes:
                for i in range(len(route)-1):
                    tau[route[i], route[i+1]] += deposit
        # Deposit on iteration best (if different from global best)
        if iteration_best_routes is not None and iteration_best_max < best_max:
            deposit = 1.0 / iteration_best_max
            for route in iteration_best_routes:
                for i in range(len(route)-1):
                    tau[route[i], route[i+1]] += deposit
    
    # If no solution found, return trivial routes (should not happen)
    if best_routes is None:
        best_routes = [[0, 0] for _ in range(truck_count)]
    return best_routes
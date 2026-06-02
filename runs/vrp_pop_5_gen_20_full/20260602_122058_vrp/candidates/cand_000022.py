import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    random.seed(0)
    
    # Trivial case: each customer gets its own truck
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # ---------- Initial construction: farthest-first seeds + regret assignment ----------
    seeds = []
    first_seed = max(customers, key=lambda i: distance_matrix[0][i])
    seeds.append(first_seed)
    for _ in range(1, truck_count):
        best_min_dist = -1
        best_node = None
        for node in customers:
            if node in seeds:
                continue
            min_dist = min(distance_matrix[node][s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and node < best_node):
                best_min_dist = min_dist
                best_node = node
        seeds.append(best_node)
    
    # Assign customers to seeds using regret (max savings) heuristic
    unassigned = set(customers) - set(seeds)
    clusters = {s: [s] for s in seeds}
    # Precompute distances from each customer to each seed
    dist_to_seed = {c: {s: distance_matrix[c][s] for s in seeds} for c in unassigned}
    
    while unassigned:
        # Compute regret for each customer: difference between best and second best seed distance
        regret = {}
        for c in unassigned:
            sorted_seeds = sorted(seeds, key=lambda s: dist_to_seed[c][s])
            best = dist_to_seed[c][sorted_seeds[0]]
            second_best = dist_to_seed[c][sorted_seeds[1]] if len(sorted_seeds) > 1 else best
            regret[c] = second_best - best
        # Select customer with max regret
        chosen = max(regret, key=lambda x: (regret[x], x))
        # Assign to nearest seed
        nearest = min(seeds, key=lambda s: (dist_to_seed[chosen][s], s))
        clusters[nearest].append(chosen)
        unassigned.remove(chosen)
    
    # Build each route using cheapest insertion from cluster
    def route_dist(r):
        d = 0
        for i in range(len(r)-1):
            d += distance_matrix[r[i]][r[i+1]]
        return d
    
    def cheapest_insertion_route(customers_in_cluster):
        # Start with depot on both ends
        route = [0, 0]
        for c in customers_in_cluster:
            best_cost = float('inf')
            best_pos = 1
            for pos in range(1, len(route)):
                cost = distance_matrix[route[pos-1]][c] + distance_matrix[c][route[pos]] - distance_matrix[route[pos-1]][route[pos]]
                if cost < best_cost - 1e-12 or (abs(cost - best_cost) < 1e-12 and c < best_pos):
                    best_cost = cost
                    best_pos = pos
            route.insert(best_pos, c)
        return route
    
    routes = []
    for s in seeds:
        route = cheapest_insertion_route(clusters[s])
        routes.append(route)
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    current_max = max(route_dist(r) for r in routes)
    best_routes = [r[:] for r in routes]
    best_max = current_max
    report_best_vrp(routes)
    
    # ---------- Simulated annealing improvement ----------
    max_iter = n * truck_count
    temp_init = 0.2 * current_max
    temp_final = 0.01 * current_max
    cooling = (temp_final / temp_init) ** (1.0 / max_iter) if max_iter > 0 else 1.0
    temp = temp_init
    
    # For restart: perturbation strength
    restart_interval = max(1, max_iter // 10)
    no_improve = 0
    
    for it in range(max_iter):
        # Choose move type: 0 = relocate, 1 = exchange
        move_type = random.randint(0, 1)
        # Select longest route
        max_idx = max(range(truck_count), key=lambda i: route_dist(routes[i]))
        long_route = routes[max_idx]
        
        if move_type == 0:
            # Relocate a customer from longest to another route
            if len(long_route) <= 2:
                continue
            pos = random.randint(1, len(long_route)-2)
            cust = long_route[pos]
            target_idx = random.choice([i for i in range(truck_count) if i != max_idx])
            target_route = routes[target_idx]
            ins_pos = random.randint(1, len(target_route))
            
            # Compute new routes
            new_long = long_route[:pos] + long_route[pos+1:]
            new_target = target_route[:ins_pos] + [cust] + target_route[ins_pos:]
            # Update routes temporarily
            old_long = long_route[:]
            old_target = target_route[:]
            routes[max_idx] = new_long
            routes[target_idx] = new_target
            new_max = max(route_dist(r) for r in routes)
            # Simulated annealing acceptance
            if new_max <= current_max + temp * random.random():
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                # Continue with new solution
            else:
                # Revert
                routes[max_idx] = old_long
                routes[target_idx] = old_target
        else:
            # Exchange customers between two routes
            idx1 = max_idx
            idx2 = random.choice([i for i in range(truck_count) if i != idx1])
            route1 = routes[idx1]
            route2 = routes[idx2]
            if len(route1) <= 2 or len(route2) <= 2:
                continue
            pos1 = random.randint(1, len(route1)-2)
            pos2 = random.randint(1, len(route2)-2)
            cust1 = route1[pos1]
            cust2 = route2[pos2]
            
            new_route1 = route1[:pos1] + [cust2] + route1[pos1+1:]
            new_route2 = route2[:pos2] + [cust1] + route2[pos2+1:]
            old1 = route1[:]
            old2 = route2[:]
            routes[idx1] = new_route1
            routes[idx2] = new_route2
            new_max = max(route_dist(r) for r in routes)
            if new_max <= current_max + temp * random.random():
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
            else:
                routes[idx1] = old1
                routes[idx2] = old2
        
        # Check for restart
        if new_max < best_max:
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= restart_interval:
            # Perturb best solution: swap a few customers deterministically
            best_routes_copy = [r[:] for r in best_routes]
            # Randomly select two routes and swap a customer
            for _ in range(3):
                i1 = random.randint(0, truck_count-1)
                i2 = random.randint(0, truck_count-1)
                if i1 == i2 or len(best_routes_copy[i1]) <= 2 or len(best_routes_copy[i2]) <= 2:
                    continue
                p1 = random.randint(1, len(best_routes_copy[i1])-2)
                p2 = random.randint(1, len(best_routes_copy[i2])-2)
                c1 = best_routes_copy[i1][p1]
                c2 = best_routes_copy[i2][p2]
                best_routes_copy[i1][p1] = c2
                best_routes_copy[i2][p2] = c1
            routes = [r[:] for r in best_routes_copy]
            current_max = max(route_dist(r) for r in routes)
            no_improve = 0
        
        # Cool down
        temp *= cooling
    
    # Final intra-route 2-opt on best routes
    for idx in range(truck_count):
        route = best_routes[idx]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    a = route[i-1]
                    b = route[i]
                    c = route[j]
                    d = route[j+1]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new_d = distance_matrix[a][c] + distance_matrix[b][d]
                    if new_d < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        report_best_vrp(best_routes)
                        break
                if improved:
                    break
    
    return best_routes
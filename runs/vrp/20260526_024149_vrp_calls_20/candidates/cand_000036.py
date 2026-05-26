import numpy as np
import random

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def compute_max_dist(routes, dm):
    return max(route_distance(r, dm) for r in routes)

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    num_customers = len(customers)
    
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    best_routes = None
    best_max = float('inf')
    
    num_attempts = 3
    for attempt in range(num_attempts):
        # Randomized savings construction
        routes = [[0, c, 0] for c in customers]
        while len(routes) > truck_count:
            # Compute savings and balance penalty for all feasible merges
            scores = []
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    r_i = routes[i]
                    r_j = routes[j]
                    if len(r_i) <= 2 or len(r_j) <= 2:
                        continue
                    for order in [0, 1]:
                        last_i = r_i[-2]
                        first_i = r_i[1]
                        last_j = r_j[-2]
                        first_j = r_j[1]
                        if order == 0:
                            saving = distance_matrix[0, last_i] + distance_matrix[0, first_j] - distance_matrix[last_i, first_j]
                        else:
                            saving = distance_matrix[0, last_j] + distance_matrix[0, first_i] - distance_matrix[last_j, first_i]
                        # Balance penalty
                        new_size = len(r_i) + len(r_j) - 4  # customers in merged route
                        avg_len = num_customers / truck_count
                        bal_penalty = abs(new_size - avg_len) / avg_len
                        score = saving - 0.1 * bal_penalty
                        scores.append((score, i, j, order))
            if not scores:
                break
            # Roulette wheel selection based on scores
            min_score = min(s[0] for s in scores)
            max_score = max(s[0] for s in scores)
            if max_score - min_score < 1e-12:
                # All scores equal, pick random
                idx = random.randrange(len(scores))
            else:
                # Shift scores to positive
                shifted = [s[0] - min_score + 1e-9 for s in scores]
                total = sum(shifted)
                r = random.uniform(0, total)
                cum = 0
                for idx, s in enumerate(scores):
                    cum += shifted[idx]
                    if r <= cum:
                        break
            _, i, j, order = scores[idx]
            # Perform merge
            if order == 0:
                new_route = routes[i][:-1] + routes[j][1:]
            else:
                new_route = routes[j][:-1] + routes[i][1:]
            # Remove the two routes and add new
            if i < j:
                del routes[j]
                del routes[i]
            else:
                del routes[i]
                del routes[j]
            routes.append(new_route)
        
        # Ensure exactly truck_count routes
        while len(routes) < truck_count:
            routes.append([0, 0])
        
        # Improvement loop (same as parent)
        local_best_routes = [list(r) for r in routes]
        local_best_max = compute_max_dist(routes, distance_matrix)
        report_best_vrp(routes)
        
        for _ in range(2):  # at most 2 full improvement cycles
            # Intra-route 2-opt
            for idx in range(len(routes)):
                route = routes[idx]
                if len(route) <= 3:
                    continue
                improved = True
                max_iter = len(route) * len(route)
                it = 0
                while improved and it < max_iter:
                    improved = False
                    it += 1
                    best_delta = 0
                    best_ij = None
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_dist = route_distance(new_route, distance_matrix)
                            old_dist = route_distance(route, distance_matrix)
                            delta = old_dist - new_dist
                            if delta > best_delta:
                                best_delta = delta
                                best_ij = (i, j)
                                improved = True
                    if improved:
                        i, j = best_ij
                        route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        routes[idx] = route
                routes[idx] = route
            report_best_vrp(routes)
            
            # Inter-route relocate/swap with perturbation
            max_iter = num_customers * truck_count * 2
            perturbation_count = 0
            max_perturbations = 5
            while True:
                improved_global = False
                for _ in range(max_iter):
                    dists = [route_distance(r, distance_matrix) for r in routes]
                    max_dist = max(dists)
                    best_improvement = 0
                    best_move = None
                    # Relocate
                    for i in range(len(routes)):
                        if len(routes[i]) <= 2:
                            continue
                        for pos in range(1, len(routes[i])-1):
                            customer = routes[i][pos]
                            for j in range(len(routes)):
                                if i == j:
                                    continue
                                for insert_pos in range(1, len(routes[j])):
                                    new_route_i = routes[i][:pos] + routes[i][pos+1:]
                                    new_route_j = routes[j][:insert_pos] + [customer] + routes[j][insert_pos:]
                                    new_dists = dists.copy()
                                    new_dists[i] = route_distance(new_route_i, distance_matrix)
                                    new_dists[j] = route_distance(new_route_j, distance_matrix)
                                    new_max = max(new_dists)
                                    improvement = max_dist - new_max
                                    if improvement > best_improvement + 1e-12:
                                        best_improvement = improvement
                                        best_move = ('reloc', i, pos, j, insert_pos)
                    # Swap
                    for i in range(len(routes)):
                        if len(routes[i]) <= 2:
                            continue
                        for pos_i in range(1, len(routes[i])-1):
                            cust_i = routes[i][pos_i]
                            for j in range(i+1, len(routes)):
                                if len(routes[j]) <= 2:
                                    continue
                                for pos_j in range(1, len(routes[j])-1):
                                    cust_j = routes[j][pos_j]
                                    new_route_i = routes[i][:pos_i] + [cust_j] + routes[i][pos_i+1:]
                                    new_route_j = routes[j][:pos_j] + [cust_i] + routes[j][pos_j+1:]
                                    new_dists = dists.copy()
                                    new_dists[i] = route_distance(new_route_i, distance_matrix)
                                    new_dists[j] = route_distance(new_route_j, distance_matrix)
                                    new_max = max(new_dists)
                                    improvement = max_dist - new_max
                                    if improvement > best_improvement + 1e-12:
                                        best_improvement = improvement
                                        best_move = ('swap', i, pos_i, j, pos_j)
                    if best_improvement > 1e-9:
                        typ, r1, p1, r2, p2 = best_move
                        if typ == 'reloc':
                            cust = routes[r1][p1]
                            routes[r1] = routes[r1][:p1] + routes[r1][p1+1:]
                            routes[r2] = routes[r2][:p2] + [cust] + routes[r2][p2:]
                        else:
                            cust_i = routes[r1][p1]
                            cust_j = routes[r2][p2]
                            routes[r1] = routes[r1][:p1] + [cust_j] + routes[r1][p1+1:]
                            routes[r2] = routes[r2][:p2] + [cust_i] + routes[r2][p2+1:]
                        improved_global = True
                        report_best_vrp(routes)
                    else:
                        break
                
                if not improved_global:
                    if perturbation_count >= max_perturbations:
                        break
                    perturbation_count += 1
                    # Find least worsening move
                    dists = [route_distance(r, distance_matrix) for r in routes]
                    max_dist = max(dists)
                    best_worsen = 1e9
                    best_move = None
                    # Relocate
                    for i in range(len(routes)):
                        if len(routes[i]) <= 2:
                            continue
                        for pos in range(1, len(routes[i])-1):
                            customer = routes[i][pos]
                            for j in range(len(routes)):
                                if i == j:
                                    continue
                                for insert_pos in range(1, len(routes[j])):
                                    new_route_i = routes[i][:pos] + routes[i][pos+1:]
                                    new_route_j = routes[j][:insert_pos] + [customer] + routes[j][insert_pos:]
                                    new_dists = dists.copy()
                                    new_dists[i] = route_distance(new_route_i, distance_matrix)
                                    new_dists[j] = route_distance(new_route_j, distance_matrix)
                                    new_max = max(new_dists)
                                    worsen = new_max - max_dist
                                    if worsen < best_worsen - 1e-12:
                                        best_worsen = worsen
                                        best_move = ('reloc', i, pos, j, insert_pos)
                    # Swap
                    for i in range(len(routes)):
                        if len(routes[i]) <= 2:
                            continue
                        for pos_i in range(1, len(routes[i])-1):
                            cust_i = routes[i][pos_i]
                            for j in range(i+1, len(routes)):
                                if len(routes[j]) <= 2:
                                    continue
                                for pos_j in range(1, len(routes[j])-1):
                                    cust_j = routes[j][pos_j]
                                    new_route_i = routes[i][:pos_i] + [cust_j] + routes[i][pos_i+1:]
                                    new_route_j = routes[j][:pos_j] + [cust_i] + routes[j][pos_j+1:]
                                    new_dists = dists.copy()
                                    new_dists[i] = route_distance(new_route_i, distance_matrix)
                                    new_dists[j] = route_distance(new_route_j, distance_matrix)
                                    new_max = max(new_dists)
                                    worsen = new_max - max_dist
                                    if worsen < best_worsen - 1e-12:
                                        best_worsen = worsen
                                        best_move = ('swap', i, pos_i, j, pos_j)
                    if best_move is None or best_worsen >= 1e9:
                        break
                    typ, r1, p1, r2, p2 = best_move
                    if typ == 'reloc':
                        cust = routes[r1][p1]
                        routes[r1] = routes[r1][:p1] + routes[r1][p1+1:]
                        routes[r2] = routes[r2][:p2] + [cust] + routes[r2][p2:]
                    else:
                        cust_i = routes[r1][p1]
                        cust_j = routes[r2][p2]
                        routes[r1] = routes[r1][:p1] + [cust_j] + routes[r1][p1+1:]
                        routes[r2] = routes[r2][:p2] + [cust_i] + routes[r2][p2+1:]
                    report_best_vrp(routes)
                else:
                    perturbation_count = 0
                    continue
                break  # after perturbation, break out of while and restart improvement loop
        
        # Post-balancing: try moving customers from longest to shortest routes if reduces max
        for _ in range(num_customers):
            dists = [route_distance(r, distance_matrix) for r in routes]
            max_idx = np.argmax(dists)
            max_dist = dists[max_idx]
            # Find a customer to move from max route to a shorter route if reduces max
            best_reduction = 0
            best_move = None
            if len(routes[max_idx]) > 2:
                for pos in range(1, len(routes[max_idx])-1):
                    cust = routes[max_idx][pos]
                    for j in range(len(routes)):
                        if j == max_idx:
                            continue
                        if len(routes[j]) <= 2:
                            continue
                        # Only consider moves that reduce max (or at least not increase)
                        for insert_pos in range(1, len(routes[j])):
                            new_route_i = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                            new_route_j = routes[j][:insert_pos] + [cust] + routes[j][insert_pos:]
                            new_dists = dists.copy()
                            new_dists[max_idx] = route_distance(new_route_i, distance_matrix)
                            new_dists[j] = route_distance(new_route_j, distance_matrix)
                            new_max = max(new_dists)
                            reduction = max_dist - new_max
                            if reduction > best_reduction:
                                best_reduction = reduction
                                best_move = (max_idx, pos, j, insert_pos)
            if best_reduction > 1e-9:
                i, pos, j, insert_pos = best_move
                cust = routes[i][pos]
                routes[i] = routes[i][:pos] + routes[i][pos+1:]
                routes[j] = routes[j][:insert_pos] + [cust] + routes[j][insert_pos:]
                report_best_vrp(routes)
            else:
                break
        
        # Update local best
        current_max = compute_max_dist(routes, distance_matrix)
        if current_max < local_best_max - 1e-12:
            local_best_max = current_max
            local_best_routes = [list(r) for r in routes]
        
        # Update global best
        if local_best_max < best_max - 1e-12:
            best_max = local_best_max
            best_routes = [list(r) for r in local_best_routes]
    
    # Return best found
    if best_routes is None:
        best_routes = [[0, c, 0] for c in customers]
        while len(best_routes) < truck_count:
            best_routes.append([0, 0])
    report_best_vrp(best_routes)
    return best_routes
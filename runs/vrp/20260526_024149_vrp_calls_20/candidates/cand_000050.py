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
    num_restarts = 10
    for restart in range(num_restarts):
        shuffled = customers[:]
        random.shuffle(shuffled)
        routes = [[0, c, 0] for c in shuffled]
        while len(routes) > truck_count:
            best_saving = -1e9
            best_trad_saving = -1e9
            best_pair = None
            best_order = 0
            current_max = compute_max_dist(routes, distance_matrix)
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    r_i = routes[i]
                    r_j = routes[j]
                    if len(r_i) <= 2 or len(r_j) <= 2:
                        continue
                    for order in [0, 1]:
                        if order == 0:
                            new_route = r_i[:-1] + r_j[1:]
                        else:
                            new_route = r_j[:-1] + r_i[1:]
                        other_dists = [route_distance(routes[k], distance_matrix) for k in range(len(routes)) if k != i and k != j]
                        new_dist = route_distance(new_route, distance_matrix)
                        new_max = max(max(other_dists) if other_dists else 0, new_dist)
                        saving = current_max - new_max
                        last_i = r_i[-2]
                        first_i = r_i[1]
                        last_j = r_j[-2]
                        first_j = r_j[1]
                        if order == 0:
                            trad_saving = distance_matrix[0, last_i] + distance_matrix[0, first_j] - distance_matrix[last_i, first_j]
                        else:
                            trad_saving = distance_matrix[0, last_j] + distance_matrix[0, first_i] - distance_matrix[last_j, first_i]
                        if saving > best_saving + 1e-12 or (abs(saving - best_saving) <= 1e-12 and trad_saving > best_trad_saving):
                            best_saving = saving
                            best_trad_saving = trad_saving
                            best_pair = (i, j)
                            best_order = order
            if best_pair is None:
                break
            i, j = best_pair
            if best_order == 0:
                new_route = routes[i][:-1] + routes[j][1:]
            else:
                new_route = routes[j][:-1] + routes[i][1:]
            if i < j:
                del routes[j]
                del routes[i]
            else:
                del routes[i]
                del routes[j]
            routes.append(new_route)
        
        report_best_vrp(routes)
        
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
                        if delta > best_delta + 1e-12:
                            best_delta = delta
                            best_ij = (i, j)
                            improved = True
                if improved:
                    i, j = best_ij
                    route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    routes[idx] = route
            routes[idx] = route
        
        report_best_vrp(routes)
        
        # Inter-route improvement and ruin-and-recreate perturbation
        max_iter = num_customers * truck_count * 2
        perturbation_count = 0
        max_perturbations = 3
        while True:
            # Improvement loop
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
                # 2-opt*
                for i in range(len(routes)):
                    if len(routes[i]) <= 3:
                        continue
                    for j in range(i+1, len(routes)):
                        if len(routes[j]) <= 3:
                            continue
                        for p1 in range(1, len(routes[i])-1):
                            for p2 in range(1, len(routes[j])-1):
                                # Option A
                                new_route_i = routes[i][:p1] + routes[j][p2:]
                                new_route_j = routes[j][:p2] + routes[i][p1:]
                                if new_route_i[0] != 0 or new_route_i[-1] != 0 or new_route_j[0] != 0 or new_route_j[-1] != 0:
                                    continue
                                new_dists = dists.copy()
                                new_dists[i] = route_distance(new_route_i, distance_matrix)
                                new_dists[j] = route_distance(new_route_j, distance_matrix)
                                new_max = max(new_dists)
                                improvement = max_dist - new_max
                                if improvement > best_improvement + 1e-12:
                                    best_improvement = improvement
                                    best_move = ('cross', i, p1, j, p2, 'A')
                if best_improvement > 1e-9:
                    typ = best_move[0]
                    if typ == 'reloc':
                        _, r1, p1, r2, p2 = best_move
                        cust = routes[r1][p1]
                        routes[r1] = routes[r1][:p1] + routes[r1][p1+1:]
                        routes[r2] = routes[r2][:p2] + [cust] + routes[r2][p2:]
                    elif typ == 'swap':
                        _, r1, p1, r2, p2 = best_move
                        cust_i = routes[r1][p1]
                        cust_j = routes[r2][p2]
                        routes[r1] = routes[r1][:p1] + [cust_j] + routes[r1][p1+1:]
                        routes[r2] = routes[r2][:p2] + [cust_i] + routes[r2][p2+1:]
                    else: # cross
                        _, r1, p1, r2, p2, _ = best_move
                        new_route_i = routes[r1][:p1] + routes[r2][p2:]
                        new_route_j = routes[r2][:p2] + routes[r1][p1:]
                        routes[r1] = new_route_i
                        routes[r2] = new_route_j
                    improved_global = True
                    report_best_vrp(routes)
                else:
                    break
            
            if not improved_global:
                # Ruin-and-recreate perturbation
                if perturbation_count >= max_perturbations:
                    break
                perturbation_count += 1
                # Randomly select 1-3 customers to remove
                all_customers = []
                for r in routes:
                    if len(r) > 2:
                        all_customers.extend(r[1:-1])
                if len(all_customers) <= 1:
                    break
                remove_count = min(random.randint(1, 3), len(all_customers) - 1)
                removed = random.sample(all_customers, remove_count)
                # Remove them from routes
                for cust in removed:
                    for idx, r in enumerate(routes):
                        if cust in r:
                            pos = r.index(cust)
                            routes[idx] = r[:pos] + r[pos+1:]
                            break
                # Reinsert each removed customer in best position (minimize max distance)
                for cust in removed:
                    best_worsen = float('inf')
                    best_insert = None
                    dists = [route_distance(r, distance_matrix) for r in routes]
                    current_max = max(dists)
                    for idx, r in enumerate(routes):
                        for insert_pos in range(1, len(r)):
                            new_r = r[:insert_pos] + [cust] + r[insert_pos:]
                            new_dist = route_distance(new_r, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[idx] = new_dist
                            new_max = max(new_dists)
                            worsen = new_max - current_max
                            if worsen < best_worsen - 1e-12:
                                best_worsen = worsen
                                best_insert = (idx, insert_pos)
                    if best_insert is not None:
                        idx, pos = best_insert
                        routes[idx] = routes[idx][:pos] + [cust] + routes[idx][pos:]
                report_best_vrp(routes)
                # After ruin-and-recreate, continue improvement loop
                continue
            else:
                perturbation_count = 0
        
        current_max = compute_max_dist(routes, distance_matrix)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
    
    if best_routes is not None:
        routes = best_routes
    report_best_vrp(routes)
    return routes
import numpy as np

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def compute_max_dist(routes, dm):
    return max(route_distance(r, dm) for r in routes)

def compute_all_dists(routes, dm):
    return [route_distance(r, dm) for r in routes]

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
    
    # Clarke-Wright construction with savings based on reduction in max route distance
    routes = [[0, c, 0] for c in customers]
    
    while len(routes) > truck_count:
        best_saving = -1e9
        best_pair = None
        best_order = 0
        best_balance = 1e9  # for tie-breaking: lower variance is better
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
                    new_max = max(other_dists + [new_dist])
                    saving = current_max - new_max
                    # compute variance of route distances if merged: include new route, exclude i and j
                    all_dists_after = other_dists + [new_dist]
                    mean_dist = np.mean(all_dists_after)
                    variance = np.var(all_dists_after)
                    # tie-break: prefer larger saving, then lower variance
                    if saving > best_saving + 1e-12:
                        best_saving = saving
                        best_pair = (i, j)
                        best_order = order
                        best_balance = variance
                    elif abs(saving - best_saving) < 1e-12:
                        if variance < best_balance - 1e-12:
                            best_balance = variance
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
                    delta = route_distance(route, distance_matrix) - route_distance(new_route, distance_matrix)
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
    
    # Inter-route improvement with adaptive perturbation
    max_perturbations = 3
    perturbation_count = 0
    stagnation = 0
    max_stagnation = 2  # if no improvement after perturbation, increase perturbation count
    
    while True:
        # Improvement loop
        improved_global = False
        max_iter = num_customers * truck_count * 2
        for _ in range(max_iter):
            dists = compute_all_dists(routes, distance_matrix)
            max_dist = max(dists)
            best_improvement = 0
            best_move = None  # (type, data)
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
            # Or-opt: relocate segments of length up to 3
            for seg_len in [2, 3]:
                for i in range(len(routes)):
                    if len(routes[i]) <= seg_len + 1:
                        continue
                    for start in range(1, len(routes[i]) - seg_len):
                        segment = routes[i][start:start+seg_len]
                        for j in range(len(routes)):
                            if i == j:
                                continue
                            for insert_pos in range(1, len(routes[j])):
                                new_route_i = routes[i][:start] + routes[i][start+seg_len:]
                                new_route_j = routes[j][:insert_pos] + segment + routes[j][insert_pos:]
                                new_dists = dists.copy()
                                new_dists[i] = route_distance(new_route_i, distance_matrix)
                                new_dists[j] = route_distance(new_route_j, distance_matrix)
                                new_max = max(new_dists)
                                improvement = max_dist - new_max
                                if improvement > best_improvement + 1e-12:
                                    best_improvement = improvement
                                    best_move = ('oropt', i, start, seg_len, j, insert_pos)
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
                else:  # oropt
                    _, r1, start, seg_len, r2, insert_pos = best_move
                    segment = routes[r1][start:start+seg_len]
                    routes[r1] = routes[r1][:start] + routes[r1][start+seg_len:]
                    routes[r2] = routes[r2][:insert_pos] + segment + routes[r2][insert_pos:]
                improved_global = True
                report_best_vrp(routes)
            else:
                break
        
        if not improved_global:
            # Perturbation: accept the least worsening move
            if perturbation_count >= max_perturbations:
                break
            perturbation_count += 1
            # Find the move that increases max distance the least
            best_worsen = 1e9
            best_move = None
            dists = compute_all_dists(routes, distance_matrix)
            max_dist = max(dists)
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
            # Or-opt segments
            for seg_len in [2, 3]:
                for i in range(len(routes)):
                    if len(routes[i]) <= seg_len + 1:
                        continue
                    for start in range(1, len(routes[i]) - seg_len):
                        segment = routes[i][start:start+seg_len]
                        for j in range(len(routes)):
                            if i == j:
                                continue
                            for insert_pos in range(1, len(routes[j])):
                                new_route_i = routes[i][:start] + routes[i][start+seg_len:]
                                new_route_j = routes[j][:insert_pos] + segment + routes[j][insert_pos:]
                                new_dists = dists.copy()
                                new_dists[i] = route_distance(new_route_i, distance_matrix)
                                new_dists[j] = route_distance(new_route_j, distance_matrix)
                                new_max = max(new_dists)
                                worsen = new_max - max_dist
                                if worsen < best_worsen - 1e-12:
                                    best_worsen = worsen
                                    best_move = ('oropt', i, start, seg_len, j, insert_pos)
            if best_move is None or best_worsen >= 1e9:
                break
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
            else:  # oropt
                _, r1, start, seg_len, r2, insert_pos = best_move
                segment = routes[r1][start:start+seg_len]
                routes[r1] = routes[r1][:start] + routes[r1][start+seg_len:]
                routes[r2] = routes[r2][:insert_pos] + segment + routes[r2][insert_pos:]
            report_best_vrp(routes)
            # Adaptive: if still no improvement after perturbation for 2 times, increase perturbation count cap
            stagnation += 1
            if stagnation >= max_stagnation:
                max_perturbations = min(max_perturbations + 1, 5)
                stagnation = 0
        else:
            stagnation = 0
            # reset perturbation count after improvement
            perturbation_count = 0
            continue
    
    report_best_vrp(routes)
    return routes
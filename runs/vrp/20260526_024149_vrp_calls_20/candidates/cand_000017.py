import numpy as np
import random

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def copy_routes(routes):
    return [r[:] for r in routes]

def solve_vrp(distance_matrix, truck_count):
    random.seed(42)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    num_customers = len(customers)
    
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Clarke-Wright construction with improved tie-breaking
    routes = [[0, c, 0] for c in customers]
    while len(routes) > truck_count:
        best_saving = -1e9
        best_dij = -1e9
        best_pair = None
        best_order = 0
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                r_i = routes[i]
                r_j = routes[j]
                if len(r_i) <= 2 or len(r_j) <= 2:
                    continue
                last_i = r_i[-2]
                first_i = r_i[1]
                last_j = r_j[-2]
                first_j = r_j[1]
                s1 = distance_matrix[0, last_i] + distance_matrix[0, first_j] - distance_matrix[last_i, first_j]
                s2 = distance_matrix[0, last_j] + distance_matrix[0, first_i] - distance_matrix[last_j, first_i]
                d1 = distance_matrix[last_i, first_j]
                d2 = distance_matrix[last_j, first_i]
                # compare (saving, dij) lexicographically
                if (s1 > best_saving) or (s1 == best_saving and d1 > best_dij):
                    best_saving = s1
                    best_dij = d1
                    best_pair = (i, j)
                    best_order = 0
                if (s2 > best_saving) or (s2 == best_saving and d2 > best_dij):
                    best_saving = s2
                    best_dij = d2
                    best_pair = (i, j)
                    best_order = 1
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
    best_routes = copy_routes(routes)
    best_max = max(route_distance(r, distance_matrix) for r in routes)
    
    # Intra-route 2-opt
    def improve_intra(routes):
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
        return routes
    
    routes = improve_intra(routes)
    report_best_vrp(routes)
    cur_max = max(route_distance(r, distance_matrix) for r in routes)
    if cur_max < best_max:
        best_routes = copy_routes(routes)
        best_max = cur_max
        report_best_vrp(best_routes)
    
    # Inter-route improvement with perturbation restart
    def improve_inter(routes):
        max_iter = num_customers * truck_count * 2
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
                            if improvement > best_improvement:
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
                            if improvement > best_improvement:
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
                report_best_vrp(routes)
            else:
                break
        return routes
    
    routes = improve_inter(routes)
    cur_max = max(route_distance(r, distance_matrix) for r in routes)
    if cur_max < best_max:
        best_routes = copy_routes(routes)
        best_max = cur_max
        report_best_vrp(best_routes)
    
    # Perturbation restart
    num_restarts = 3
    for restart in range(num_restarts):
        # Shake: randomly relocate a few customers
        shake_routes = copy_routes(best_routes)
        shake_moves = min(5, num_customers // 2)
        for _ in range(shake_moves):
            # pick a random route with at least one customer
            candidate_routes = [i for i, r in enumerate(shake_routes) if len(r) > 2]
            if not candidate_routes:
                break
            i = random.choice(candidate_routes)
            pos = random.randint(1, len(shake_routes[i])-2)
            cust = shake_routes[i][pos]
            # remove customer
            shake_routes[i] = shake_routes[i][:pos] + shake_routes[i][pos+1:]
            # insert into another random route (possibly same? no, different)
            j = random.choice([idx for idx in range(len(shake_routes)) if idx != i])
            insert_pos = random.randint(1, len(shake_routes[j]))
            shake_routes[j] = shake_routes[j][:insert_pos] + [cust] + shake_routes[j][insert_pos:]
        # Re-optimize
        shake_routes = improve_intra(shake_routes)
        shake_routes = improve_inter(shake_routes)
        cur_max = max(route_distance(r, distance_matrix) for r in shake_routes)
        if cur_max < best_max:
            best_routes = copy_routes(shake_routes)
            best_max = cur_max
            report_best_vrp(best_routes)
    
    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.insert(0, [0,0])
    if len(best_routes) > truck_count:
        # should not happen, but merge excess routes into one
        excess = best_routes[-1]
        best_routes = best_routes[:-1]
        if len(excess) > 2:
            best_routes[-1] = best_routes[-1][:-1] + excess[1:]
    report_best_vrp(best_routes)
    return best_routes
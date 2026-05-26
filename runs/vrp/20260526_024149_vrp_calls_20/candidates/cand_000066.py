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
    num_restarts = 8
    for restart in range(num_restarts):
        # Random initial assignment
        shuffled = customers[:]
        random.shuffle(shuffled)
        # Initialize empty routes
        routes = [[] for _ in range(truck_count)]
        # Distribute customers randomly among trucks
        for c in shuffled:
            t = random.randint(0, truck_count-1)
            routes[t].append(c)
        # For empty routes, just [0,0]
        for i in range(truck_count):
            if len(routes[i]) == 0:
                routes[i] = [0, 0]
            else:
                # Random order within route
                random.shuffle(routes[i])
                routes[i] = [0] + routes[i] + [0]

        # Ensure we have exactly truck_count routes (some may be empty)
        while len(routes) < truck_count:
            routes.append([0, 0])

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

        max_iter_improve = num_customers * truck_count * 2
        perturbation_count = 0
        max_perturbations = 5
        while True:
            improved_global = False
            for _ in range(max_iter_improve):
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
                # 2-opt* (cross exchange)
                for i in range(len(routes)):
                    if len(routes[i]) <= 3:
                        continue
                    for j in range(i+1, len(routes)):
                        if len(routes[j]) <= 3:
                            continue
                        for p1 in range(1, len(routes[i])-1):
                            for p2 in range(1, len(routes[j])-1):
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
                                    best_move = ('cross', i, p1, j, p2)
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
                    else:
                        _, r1, p1, r2, p2 = best_move
                        new_route_i = routes[r1][:p1] + routes[r2][p2:]
                        new_route_j = routes[r2][:p2] + routes[r1][p1:]
                        routes[r1] = new_route_i
                        routes[r2] = new_route_j
                    improved_global = True
                    report_best_vrp(routes)
                else:
                    break

            if not improved_global:
                if perturbation_count >= max_perturbations:
                    break
                perturbation_count += 1
                # Shaking phase: random swaps between routes (diversification)
                num_shakes = min(5, num_customers)
                for _ in range(num_shakes):
                    # Pick two distinct routes with customers
                    nonempty = [i for i, r in enumerate(routes) if len(r) > 2]
                    if len(nonempty) < 2:
                        break
                    i, j = random.sample(nonempty, 2)
                    if len(routes[i]) <= 2 or len(routes[j]) <= 2:
                        continue
                    # Random positions for customers (excluding depot)
                    pos_i = random.randint(1, len(routes[i])-2)
                    pos_j = random.randint(1, len(routes[j])-2)
                    cust_i = routes[i][pos_i]
                    cust_j = routes[j][pos_j]
                    # Swap
                    routes[i][pos_i] = cust_j
                    routes[j][pos_j] = cust_i
                report_best_vrp(routes)

                # Ejection perturbation (same as parent but maybe more aggressive?)
                current_customers = []
                for r in routes:
                    for c in r[1:-1]:
                        current_customers.append(c)
                if len(current_customers) < 3:
                    break
                k = random.randint(1, min(5, len(current_customers)))
                ejected = random.sample(current_customers, k)
                for r_idx in range(len(routes)):
                    route = routes[r_idx]
                    for cust in ejected:
                        while cust in route:
                            route.remove(cust)
                    if len(route) == 2:
                        routes[r_idx] = [0, 0]
                    else:
                        routes[r_idx] = route
                # Greedy reinsertion to minimize max distance
                for cust in ejected:
                    best_route_idx = None
                    best_pos = None
                    best_new_max = float('inf')
                    for r_idx in range(len(routes)):
                        route = routes[r_idx]
                        for pos in range(1, len(route)):
                            new_route = route[:pos] + [cust] + route[pos:]
                            old_dist = route_distance(route, distance_matrix)
                            new_dist = route_distance(new_route, distance_matrix)
                            other_dists = [route_distance(routes[j], distance_matrix) for j in range(len(routes)) if j != r_idx]
                            new_max = max(max(other_dists) if other_dists else 0, new_dist)
                            if new_max < best_new_max - 1e-12:
                                best_new_max = new_max
                                best_route_idx = r_idx
                                best_pos = pos
                    if best_route_idx is not None:
                        routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
                report_best_vrp(routes)
            else:
                perturbation_count = 0
                continue

        current_max = compute_max_dist(routes, distance_matrix)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]

    if best_routes is not None:
        routes = best_routes
    report_best_vrp(routes)
    return routes
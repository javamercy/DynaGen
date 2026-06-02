import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        routes += [[0, 0]] * (truck_count - (n - 1))
        return routes
    
    def route_distance(route):
        dist = 0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        return dist
    
    def compute_distances(routes):
        return [route_distance(r) for r in routes]
    
    def copy_routes(routes):
        return [r[:] for r in routes]
    
    # Farthest-point seeding
    seed_customers = []
    first_seed = max(range(1, n), key=lambda c: distance_matrix[0][c])
    seed_customers.append(first_seed)
    while len(seed_customers) < truck_count:
        best_cust = None
        best_min_dist = -1
        for c in range(1, n):
            if c in seed_customers:
                continue
            min_dist = min(distance_matrix[c][s] for s in seed_customers)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_cust = c
        seed_customers.append(best_cust)
    
    routes = [[0, s, 0] for s in seed_customers]
    route_dists = [distance_matrix[0][s] + distance_matrix[s][0] for s in seed_customers]
    assigned = set(seed_customers)
    remaining = [c for c in range(1, n) if c not in assigned]
    
    # Regret insertion
    while remaining:
        best_regret = -1
        best_cust = None
        best_route_idx = None
        best_pos = None
        for cust in remaining:
            insertion_costs = []
            for idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    new_dist = route_dists[idx] - distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                    overall_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                    insertion_costs.append((overall_max, idx, pos))
            insertion_costs.sort(key=lambda x: (x[0], x[1], x[2]))
            best_cost = insertion_costs[0][0]
            if len(insertion_costs) > 1:
                second_cost = insertion_costs[1][0]
                regret = second_cost - best_cost
            else:
                regret = best_cost
            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_new_max, best_route_idx, best_pos = insertion_costs[0]
        
        route = routes[best_route_idx]
        pos = best_pos
        prev = route[pos-1]
        nxt = route[pos]
        route_dists[best_route_idx] += -distance_matrix[prev][nxt] + distance_matrix[prev][best_cust] + distance_matrix[best_cust][nxt]
        route.insert(pos, best_cust)
        assigned.add(best_cust)
        remaining.remove(best_cust)
    
    best_routes = copy_routes(routes)
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    max_iter = max(30, n)
    stagnation = 0
    max_stagnation = 5
    restart_count = 0
    max_restarts = 2
    
    for _ in range(max_iter):
        improved = False
        
        # Limited best-improvement: random subset of moves
        # Intra 2-opt
        for idx, route in enumerate(routes):
            if len(route) <= 4:
                continue
            candidates = []
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j == i+1:
                        continue
                    candidates.append((i,j))
            if not candidates:
                continue
            # sample up to 50 candidates
            sample_size = min(50, len(candidates))
            selected = random.sample(candidates, sample_size)
            best_candidate = None
            best_new_dist = None
            for (i,j) in selected:
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_dist = route_distance(new_route)
                new_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                if best_candidate is None or new_max < best_new_max:
                    best_candidate = (i,j, new_route, new_dist, new_max)
                    best_new_max = new_max
            if best_candidate and best_new_max < max(route_dists):
                routes[idx] = best_candidate[2]
                route_dists[idx] = best_candidate[3]
                improved = True
                break
        if improved:
            if max(route_dists) < best_max:
                best_routes = copy_routes(routes)
                best_max = max(route_dists)
                report_best_vrp(best_routes)
            stagnation = 0
            continue
        
        # Inter relocate
        for i in range(truck_count):
            route_i = routes[i]
            if len(route_i) <= 3:
                continue
            candidates = []
            for pos_i in range(1, len(route_i)-1):
                cust = route_i[pos_i]
                new_i = route_i[:pos_i] + route_i[pos_i+1:]
                new_i_dist = route_distance(new_i)
                for j in range(truck_count):
                    if i == j:
                        continue
                    route_j = routes[j]
                    for pos_j in range(1, len(route_j)):
                        prev = route_j[pos_j-1]
                        nxt = route_j[pos_j]
                        inc = -distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                        new_j_dist = route_dists[j] + inc
                        overall_max = max(route_dists[:i] + [new_i_dist] + route_dists[i+1:j] + [new_j_dist] + route_dists[j+1:])
                        candidates.append((overall_max, i, j, pos_i, pos_j, new_i, new_i_dist, new_j_dist))
            if not candidates:
                continue
            sample_size = min(50, len(candidates))
            selected = random.sample(candidates, sample_size)
            best = min(selected, key=lambda x: x[0])
            if best[0] < max(route_dists):
                overall_max, i, j, pos_i, pos_j, new_i, new_i_dist, new_j_dist = best
                # reconstruct new_j
                cust = routes[i][pos_i]
                route_j = routes[j]
                new_j = route_j[:pos_j] + [cust] + route_j[pos_j:]
                routes[i] = new_i
                routes[j] = new_j
                route_dists[i] = new_i_dist
                route_dists[j] = new_j_dist
                improved = True
                break
        if improved:
            if max(route_dists) < best_max:
                best_routes = copy_routes(routes)
                best_max = max(route_dists)
                report_best_vrp(best_routes)
            stagnation = 0
            continue
        
        # Inter exchange
        for i in range(truck_count):
            route_i = routes[i]
            if len(route_i) <= 3:
                continue
            for pos_i in range(1, len(route_i)-1):
                cust_i = route_i[pos_i]
                for j in range(i+1, truck_count):
                    route_j = routes[j]
                    if len(route_j) <= 3:
                        continue
                    for pos_j in range(1, len(route_j)-1):
                        cust_j = route_j[pos_j]
                        candidates = []
                        # Remove both
                        new_i = route_i[:pos_i] + route_i[pos_i+1:]
                        new_j = route_j[:pos_j] + route_j[pos_j+1:]
                        # Insert cust_i into j at best position
                        best_inc_j = float('inf')
                        best_pos_ji = -1
                        for p in range(1, len(new_j)):
                            prev = new_j[p-1]
                            nxt = new_j[p]
                            inc = -distance_matrix[prev][nxt] + distance_matrix[prev][cust_i] + distance_matrix[cust_i][nxt]
                            if inc < best_inc_j:
                                best_inc_j = inc
                                best_pos_ji = p
                        if best_pos_ji == -1:
                            continue
                        final_j = new_j[:best_pos_ji] + [cust_i] + new_j[best_pos_ji:]
                        final_j_dist = route_distance(final_j)
                        # Insert cust_j into i at best position
                        best_inc_i = float('inf')
                        best_pos_ij = -1
                        for p in range(1, len(new_i)):
                            prev = new_i[p-1]
                            nxt = new_i[p]
                            inc = -distance_matrix[prev][nxt] + distance_matrix[prev][cust_j] + distance_matrix[cust_j][nxt]
                            if inc < best_inc_i:
                                best_inc_i = inc
                                best_pos_ij = p
                        if best_pos_ij == -1:
                            continue
                        final_i = new_i[:best_pos_ij] + [cust_j] + new_i[best_pos_ij:]
                        final_i_dist = route_distance(final_i)
                        overall_max = max(route_dists[:i] + [final_i_dist] + route_dists[i+1:j] + [final_j_dist] + route_dists[j+1:])
                        candidates.append((overall_max, final_i, final_i_dist, final_j, final_j_dist, i, j))
                        if not candidates:
                            continue
                        # only one candidate per (i,j,pos_i,pos_j) due to best insertion positions
                        best_candidate = min(candidates, key=lambda x: x[0])
                        if best_candidate[0] < max(route_dists):
                            overall_max, final_i, final_i_dist, final_j, final_j_dist, i, j = best_candidate
                            routes[i] = final_i
                            routes[j] = final_j
                            route_dists[i] = final_i_dist
                            route_dists[j] = final_j_dist
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            if max(route_dists) < best_max:
                best_routes = copy_routes(routes)
                best_max = max(route_dists)
                report_best_vrp(best_routes)
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= max_stagnation and restart_count < max_restarts:
                # Restart: remove a fraction and reinsert with regret
                num_remove = max(1, int(0.1 * (n-1)))
                all_assigned = list(assigned)
                if len(all_assigned) <= num_remove:
                    break
                remove_set = set(random.sample(all_assigned, num_remove))
                new_routes = []
                new_assigned = set()
                for route in routes:
                    new_route = [0]
                    for node in route[1:-1]:
                        if node not in remove_set:
                            new_route.append(node)
                            new_assigned.add(node)
                    new_route.append(0)
                    if len(new_route) > 2:
                        new_routes.append(new_route)
                    else:
                        new_routes.append([0,0])
                while len(new_routes) < truck_count:
                    new_routes.append([0,0])
                routes = new_routes
                route_dists = compute_distances(routes)
                assigned = new_assigned
                remaining = [c for c in range(1, n) if c not in assigned]
                # Reinsert with regret
                while remaining:
                    best_regret = -1
                    best_cust = None
                    best_route_idx = None
                    best_pos = None
                    for cust in remaining:
                        insertion_costs = []
                        for idx, route in enumerate(routes):
                            for pos in range(1, len(route)):
                                prev = route[pos-1]
                                nxt = route[pos]
                                new_dist = route_dists[idx] - distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                                overall_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                                insertion_costs.append((overall_max, idx, pos))
                        insertion_costs.sort(key=lambda x: (x[0], x[1], x[2]))
                        best_cost = insertion_costs[0][0]
                        if len(insertion_costs) > 1:
                            regret = insertion_costs[1][0] - best_cost
                        else:
                            regret = best_cost
                        if regret > best_regret:
                            best_regret = regret
                            best_cust = cust
                            best_new_max, best_route_idx, best_pos = insertion_costs[0]
                    route = routes[best_route_idx]
                    pos = best_pos
                    prev = route[pos-1]
                    nxt = route[pos]
                    route_dists[best_route_idx] += -distance_matrix[prev][nxt] + distance_matrix[prev][best_cust] + distance_matrix[best_cust][nxt]
                    route.insert(pos, best_cust)
                    assigned.add(best_cust)
                    remaining.remove(best_cust)
                current_max = max(route_dists)
                if current_max < best_max:
                    best_routes = copy_routes(routes)
                    best_max = current_max
                    report_best_vrp(best_routes)
                restart_count += 1
                stagnation = 0
    
    return best_routes[:truck_count]
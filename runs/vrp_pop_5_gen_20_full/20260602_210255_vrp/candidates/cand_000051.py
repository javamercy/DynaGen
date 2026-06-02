import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Helper functions
    def route_distance(route):
        dist = 0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        return dist
    
    def compute_max(routes):
        return max(route_distance(r) for r in routes)
    
    # Build initial routes using farthest-point seeds and regret insertion
    def build_initial():
        if truck_count >= n - 1:
            routes = [[0, c, 0] for c in customers]
            while len(routes) < truck_count:
                routes.append([0, 0])
            return routes
        
        # Farthest-point seeds
        seeds = []
        farthest = max(customers, key=lambda c: distance_matrix[0][c])
        seeds.append(farthest)
        while len(seeds) < truck_count:
            best_cust = None
            best_min_dist = -1
            for c in customers:
                if c in seeds:
                    continue
                min_dist = min(distance_matrix[c][s] for s in seeds)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_cust = c
            seeds.append(best_cust)
        
        routes = [[0, s, 0] for s in seeds]
        route_dists = [distance_matrix[0][s] + distance_matrix[s][0] for s in seeds]
        assigned = set(seeds)
        
        # Regret insertion
        unassigned = set(customers) - assigned
        while unassigned:
            best_regret = -float('inf')
            best_cust = None
            best_insertions = []
            for cust in unassigned:
                candidate_costs = []
                for idx, route in enumerate(routes):
                    best_cost = float('inf')
                    best_pos = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        new_dist = route_dists[idx] - distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                        new_max = max(max(route_dists[:idx] + [new_dist] + route_dists[idx+1:]), 0)
                        if new_max < best_cost:
                            best_cost = new_max
                            best_pos = pos
                    candidate_costs.append((best_cost, idx, best_pos))
                candidate_costs.sort(key=lambda x: x[0])
                if len(candidate_costs) == 1:
                    regret = 0
                else:
                    regret = candidate_costs[1][0] - candidate_costs[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_insertions = candidate_costs
            best_cost, idx, pos = best_insertions[0]
            route = routes[idx]
            prev = route[pos-1]
            nxt = route[pos]
            route_dists[idx] += -distance_matrix[prev][nxt] + distance_matrix[prev][best_cust] + distance_matrix[best_cust][nxt]
            route.insert(pos, best_cust)
            assigned.add(best_cust)
            unassigned.remove(best_cust)
        
        while len(routes) < truck_count:
            routes.append([0, 0])
            route_dists.append(0)
        return routes
    
    # First-improvement local search
    def improve(routes, max_iter):
        route_dists = [route_distance(r) for r in routes]
        current_max = compute_max(routes)
        best_routes = [r[:] for r in routes]
        best_max = current_max
        iteration = 0
        improved = True
        while improved and iteration < max_iter:
            improved = False
            # 2-opt
            for idx, route in enumerate(routes):
                if len(route) <= 4:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        if j == i+1:
                            continue
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        new_max_candidate = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                        if new_max_candidate < current_max:
                            routes[idx] = new_route
                            route_dists[idx] = new_dist
                            current_max = new_max_candidate
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                iteration += 1
                continue
            # Relocate
            for i in range(truck_count):
                for j in range(truck_count):
                    if i == j:
                        continue
                    route_i = routes[i]
                    route_j = routes[j]
                    if len(route_i) <= 3:
                        continue
                    for pos_i in range(1, len(route_i)-1):
                        cust = route_i[pos_i]
                        new_i = route_i[:pos_i] + route_i[pos_i+1:]
                        new_i_dist = route_distance(new_i)
                        for pos_j in range(1, len(route_j)+1):
                            temp_j = route_j[:pos_j] + [cust] + route_j[pos_j:]
                            new_j_dist = route_distance(temp_j)
                            new_max_candidate = max(route_dists[:i] + [new_i_dist] + route_dists[i+1:j] + [new_j_dist] + route_dists[j+1:])
                            if new_max_candidate < current_max:
                                routes[i] = new_i
                                route_dists[i] = new_i_dist
                                routes[j] = temp_j
                                route_dists[j] = new_j_dist
                                current_max = new_max_candidate
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                iteration += 1
                continue
            # Exchange
            for i in range(truck_count):
                for j in range(i+1, truck_count):
                    route_i = routes[i]
                    route_j = routes[j]
                    if len(route_i) <= 3 or len(route_j) <= 3:
                        continue
                    for pos_i in range(1, len(route_i)-1):
                        cust_i = route_i[pos_i]
                        for pos_j in range(1, len(route_j)-1):
                            cust_j = route_j[pos_j]
                            # Remove both
                            new_i = route_i[:pos_i] + route_i[pos_i+1:]
                            new_j = route_j[:pos_j] + route_j[pos_j+1:]
                            # Insert cust_i into best position in new_j
                            best_pos_j = -1
                            best_inc_j = float('inf')
                            for p in range(1, len(new_j)+1):
                                temp_j = new_j[:p] + [cust_i] + new_j[p:]
                                dist_j = route_distance(temp_j)
                                if dist_j < best_inc_j:
                                    best_inc_j = dist_j
                                    best_pos_j = p
                            final_j = new_j[:best_pos_j] + [cust_i] + new_j[best_pos_j:]
                            final_j_dist = best_inc_j
                            # Insert cust_j into best position in new_i
                            best_pos_i = -1
                            best_inc_i = float('inf')
                            for p in range(1, len(new_i)+1):
                                temp_i = new_i[:p] + [cust_j] + new_i[p:]
                                dist_i = route_distance(temp_i)
                                if dist_i < best_inc_i:
                                    best_inc_i = dist_i
                                    best_pos_i = p
                            final_i = new_i[:best_pos_i] + [cust_j] + new_i[best_pos_i:]
                            final_i_dist = best_inc_i
                            new_max_candidate = max(route_dists[:i] + [final_i_dist] + route_dists[i+1:j] + [final_j_dist] + route_dists[j+1:])
                            if new_max_candidate < current_max:
                                routes[i] = final_i
                                route_dists[i] = final_i_dist
                                routes[j] = final_j
                                route_dists[j] = final_j_dist
                                current_max = new_max_candidate
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                iteration += 1
        return best_routes, best_max
    
    # Main
    routes = build_initial()
    best_routes = [r[:] for r in routes]
    best_max = compute_max(routes)
    report_best_vrp(best_routes)
    
    max_iter_per_restart = n * truck_count
    num_restarts = 5
    for restart in range(num_restarts):
        if restart > 0:
            # Perturbation: randomly relocate 20% of customers
            num_to_perturb = max(1, int(n * 0.2))
            for _ in range(num_to_perturb):
                non_empty = [idx for idx, r in enumerate(routes) if len(r) > 2]
                if not non_empty:
                    continue
                src = random.choice(non_empty)
                route_src = routes[src]
                if len(route_src) <= 3:
                    continue
                pos_src = random.randint(1, len(route_src)-2)
                cust = route_src.pop(pos_src)
                dst = random.randint(0, truck_count-1)
                route_dst = routes[dst]
                if len(route_dst) == 2:
                    pos_dst = 1
                else:
                    pos_dst = random.randint(1, len(route_dst)-1)
                route_dst.insert(pos_dst, cust)
        new_routes, new_max = improve(routes, max_iter_per_restart)
        if new_max < best_max:
            best_max = new_max
            best_routes = [r[:] for r in new_routes]
            report_best_vrp(best_routes)
        routes = [r[:] for r in best_routes]  # reset to best for next restart
    return best_routes
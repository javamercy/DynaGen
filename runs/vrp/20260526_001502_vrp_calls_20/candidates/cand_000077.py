import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def insert_cost(route, node):
        best_cost = float('inf')
        best_pos = -1
        for i in range(1, len(route)):
            cost = distance_matrix[route[i-1], node] + distance_matrix[node, route[i]] - distance_matrix[route[i-1], route[i]]
            if cost < best_cost:
                best_cost = cost
                best_pos = i
        return best_cost, best_pos
    
    global_best_max = float('inf')
    global_best_routes = None
    num_restarts = 20
    no_improve_restarts = 0
    
    for restart in range(num_restarts):
        customers = list(range(1, n))
        if restart >= 5 and no_improve_restarts >= 5:
            random.shuffle(customers)
        else:
            random.shuffle(customers)
        
        routes = [[0, 0] for _ in range(truck_count)]
        current_dist = [0.0 for _ in range(truck_count)]
        unassigned = customers[:]
        lambda_balance = 0.5
        
        # Regret-2 with squared imbalance penalty
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_route_idx = None
            best_pos = None
            best_cost_val = None
            max_dist = max(current_dist)
            for cust in unassigned:
                costs_info = []
                for r in range(truck_count):
                    base_cost, pos = insert_cost(routes[r], cust)
                    # normalized squared penalty: penalty = lambda * ((max - current)/max)^2
                    if max_dist > 0:
                        penalty = lambda_balance * ((max_dist - current_dist[r]) / max_dist) ** 2
                    else:
                        penalty = 0.0
                    total_cost = base_cost + penalty
                    costs_info.append((total_cost, r, pos, base_cost))
                costs_info.sort(key=lambda x: x[0])
                best_cost = costs_info[0][0]
                second_best = costs_info[1][0] if len(costs_info) >= 2 else best_cost
                regret = second_best - best_cost
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = costs_info[0][1]
                    best_pos = costs_info[0][2]
                    best_cost_val = costs_info[0][3]
            routes[best_route_idx].insert(best_pos, best_cust)
            current_dist[best_route_idx] += best_cost_val
            unassigned.remove(best_cust)
            # Update lambda using relative imbalance
            max_dist = max(current_dist)
            avg_dist = sum(current_dist) / truck_count
            imbalance = (max_dist - avg_dist) / max_dist if max_dist > 0 else 0.0
            lambda_balance = min(1.0, max(0.1, 0.5 * (1 + imbalance)))
        
        # Post-construction imbalance reduction: relocate from longest to shortest if reduces max
        n_cust = n - 1
        improve_iter = 0
        max_improve_iters = n_cust * truck_count
        while improve_iter < max_improve_iters:
            improve_iter += 1
            max_idx = np.argmax(current_dist)
            min_idx = np.argmin(current_dist)
            if max_idx == min_idx:
                break
            route_max = routes[max_idx]
            route_min = routes[min_idx]
            if len(route_max) <= 2:
                break
            # try to find a customer in route_max that moving to route_min reduces max
            best_new_max = current_dist[max_idx]
            best_cust = None
            best_pos = None
            best_new_dist_max = None
            best_new_dist_min = None
            for idx in range(1, len(route_max)-1):
                cust = route_max[idx]
                new_route_max = route_max[:idx] + route_max[idx+1:]
                new_dist_max = route_distance(new_route_max)
                cost_min, pos_min = insert_cost(route_min, cust)
                new_dist_min = current_dist[min_idx] + cost_min
                new_max = max(new_dist_max, new_dist_min, max([current_dist[i] for i in range(truck_count) if i not in (max_idx, min_idx)]))
                if new_max < current_dist[max_idx]:  # strictly improves max
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_cust = cust
                        best_pos = pos_min
                        best_new_dist_max = new_dist_max
                        best_new_dist_min = new_dist_min
            if best_cust is not None:
                # perform move
                routes[max_idx].remove(best_cust)
                routes[max_idx] = routes[max_idx]  # already modified
                routes[min_idx].insert(best_pos, best_cust)
                current_dist[max_idx] = best_new_dist_max
                current_dist[min_idx] = best_new_dist_min
            else:
                break  # no improvement possible
        
        best_max = max(current_dist)
        local_best_routes = [list(r) for r in routes]
        local_best_max = best_max
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [list(r) for r in routes]
            report_best_vrp(routes)
            no_improve_restarts = 0
        else:
            no_improve_restarts += 1
        
        # Local search with shake (unchanged from parent)
        shake_max = 3
        shake_count = 0
        while shake_count <= shake_max:
            max_iters = 10 * n_cust * truck_count
            improved = True
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1

                # Relocate
                for r1 in range(truck_count):
                    route1 = routes[r1]
                    if len(route1) <= 2:
                        continue
                    for idx in range(1, len(route1)-1):
                        cust = route1[idx]
                        new_route1 = route1[:idx] + route1[idx+1:]
                        new_dist1 = route_distance(new_route1)
                        for r2 in range(truck_count):
                            if r2 == r1:
                                continue
                            route2 = routes[r2]
                            cost, pos = insert_cost(route2, cust)
                            new_dist2 = current_dist[r2] + cost
                            other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max:
                                routes[r1] = new_route1
                                routes[r2] = route2[:pos] + [cust] + route2[pos:]
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                best_max = new_max
                                if new_max < local_best_max:
                                    local_best_max = new_max
                                    local_best_routes = [list(r) for r in routes]
                                if new_max < global_best_max:
                                    global_best_max = new_max
                                    global_best_routes = [list(r) for r in routes]
                                    report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue

                # Swap
                for r1 in range(truck_count):
                    route1 = routes[r1]
                    if len(route1) <= 2:
                        continue
                    for idx1 in range(1, len(route1)-1):
                        cust1 = route1[idx1]
                        for r2 in range(r1+1, truck_count):
                            route2 = routes[r2]
                            if len(route2) <= 2:
                                continue
                            for idx2 in range(1, len(route2)-1):
                                cust2 = route2[idx2]
                                new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                                new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                                new_dist1 = route_distance(new_route1)
                                new_dist2 = route_distance(new_route2)
                                other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                                new_max = max(new_dist1, new_dist2, *other_dists)
                                if new_max < best_max:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    current_dist[r1] = new_dist1
                                    current_dist[r2] = new_dist2
                                    best_max = new_max
                                    if new_max < local_best_max:
                                        local_best_max = new_max
                                        local_best_routes = [list(r) for r in routes]
                                    if new_max < global_best_max:
                                        global_best_max = new_max
                                        global_best_routes = [list(r) for r in routes]
                                        report_best_vrp(routes)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue

                # Intra-route 2-opt
                for r in range(truck_count):
                    route = routes[r]
                    if len(route) <= 3:
                        continue
                    best_improve = 0
                    best_i = best_j = -1
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_dist = route_distance(new_route)
                            if new_dist < current_dist[r] - 1e-9:
                                improvement = current_dist[r] - new_dist
                                if improvement > best_improve:
                                    best_improve = improvement
                                    best_i, best_j = i, j
                    if best_improve > 0:
                        new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                        routes[r] = new_route
                        current_dist[r] = route_distance(new_route)
                        new_max = max(current_dist)
                        if new_max < best_max:
                            best_max = new_max
                            if new_max < local_best_max:
                                local_best_max = new_max
                                local_best_routes = [list(r) for r in routes]
                            if new_max < global_best_max:
                                global_best_max = new_max
                                global_best_routes = [list(r) for r in routes]
                                report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    continue

                # Cross-route 2-opt
                for r1 in range(truck_count):
                    route1 = routes[r1]
                    if len(route1) <= 2:
                        continue
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 2:
                            continue
                        for i in range(1, len(route1)-1):
                            for j in range(1, len(route2)-1):
                                new1 = route1[:i+1] + route2[j+1:]
                                new2 = route2[:j+1] + route1[i+1:]
                                new_dist1 = route_distance(new1)
                                new_dist2 = route_distance(new2)
                                other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                                new_max = max(new_dist1, new_dist2, *other_dists)
                                if new_max < best_max:
                                    routes[r1] = new1
                                    routes[r2] = new2
                                    current_dist[r1] = new_dist1
                                    current_dist[r2] = new_dist2
                                    best_max = new_max
                                    if new_max < local_best_max:
                                        local_best_max = new_max
                                        local_best_routes = [list(r) for r in routes]
                                    if new_max < global_best_max:
                                        global_best_max = new_max
                                        global_best_routes = [list(r) for r in routes]
                                        report_best_vrp(routes)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            # Local search ended
            # Check if improvement made in this restart
            if best_max < local_best_max - 1e-9:
                local_best_max = best_max
                local_best_routes = [list(r) for r in routes]
                shake_count = 0
            else:
                if shake_count < shake_max:
                    # Shake: random relocate one customer
                    all_customers = [i for i in range(1, n)]
                    random.shuffle(all_customers)
                    for cust in all_customers:
                        for r_idx, route in enumerate(routes):
                            if cust in route:
                                src_route = r_idx
                                break
                        if len(routes[src_route]) <= 2:
                            continue
                        dest_route = random.choice([r for r in range(truck_count) if r != src_route])
                        new_src = routes[src_route][:]
                        new_src.remove(cust)
                        new_dist_src = route_distance(new_src)
                        cost_dest, pos_dest = insert_cost(routes[dest_route], cust)
                        new_dest = routes[dest_route][:pos_dest] + [cust] + routes[dest_route][pos_dest:]
                        new_dist_dest = current_dist[dest_route] + cost_dest
                        routes[src_route] = new_src
                        routes[dest_route] = new_dest
                        current_dist[src_route] = new_dist_src
                        current_dist[dest_route] = new_dist_dest
                        best_max = max(current_dist)
                        break
                    shake_count += 1
                else:
                    break

    return global_best_routes
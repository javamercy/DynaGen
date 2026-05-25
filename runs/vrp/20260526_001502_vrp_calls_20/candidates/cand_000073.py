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
    
    # Regret-2 construction with balancing penalty
    def construct_routes(customers_perm):
        routes = [[0, 0] for _ in range(truck_count)]
        current_dist = [0.0 for _ in range(truck_count)]
        unassigned = customers_perm[:]
        lambda_balance = 0.5
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_route_idx = None
            best_pos = None
            best_cost_val = None
            for cust in unassigned:
                costs_info = []
                for r in range(truck_count):
                    base_cost, pos = insert_cost(routes[r], cust)
                    penalty = lambda_balance * (max(current_dist) - current_dist[r])
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
            max_dist = max(current_dist)
            avg_dist = sum(current_dist) / truck_count
            imbalance = max_dist - avg_dist
            lambda_balance = min(1.0, max(0.1, imbalance / max(avg_dist, 1e-9)))
        return routes, current_dist
    
    # Local search (first improvement)
    def local_search(routes, current_dist, best_max_in, global_best_info):
        best_max = best_max_in
        improved = True
        n_cust = n - 1
        max_iters = 10 * n_cust * truck_count
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
                        if new_max < best_max - 1e-12:
                            routes[r1] = new_route1
                            routes[r2] = route2[:pos] + [cust] + route2[pos:]
                            current_dist[r1] = new_dist1
                            current_dist[r2] = new_dist2
                            best_max = new_max
                            if best_max < global_best_info['best_max']:
                                global_best_info['best_max'] = best_max
                                global_best_info['routes'] = [list(r) for r in routes]
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
                            if new_max < best_max - 1e-12:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                best_max = new_max
                                if best_max < global_best_info['best_max']:
                                    global_best_info['best_max'] = best_max
                                    global_best_info['routes'] = [list(r) for r in routes]
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
                best_improve = 0.0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < current_dist[r] - 1e-12:
                            improvement = current_dist[r] - new_dist
                            if improvement > best_improve:
                                best_improve = improvement
                                best_i, best_j = i, j
                if best_improve > 0:
                    new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    routes[r] = new_route
                    current_dist[r] = route_distance(new_route)
                    new_max = max(current_dist)
                    if new_max < best_max - 1e-12:
                        best_max = new_max
                        if best_max < global_best_info['best_max']:
                            global_best_info['best_max'] = best_max
                            global_best_info['routes'] = [list(r) for r in routes]
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
                            other_dists = [current_dist[k] for k in range(truck_count) if k not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-12:
                                routes[r1] = new1
                                routes[r2] = new2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                best_max = new_max
                                if best_max < global_best_info['best_max']:
                                    global_best_info['best_max'] = best_max
                                    global_best_info['routes'] = [list(r) for r in routes]
                                    report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return best_max, routes, current_dist
    
    # Shake: remove customers from longest and shortest routes, then reinsert
    def shake(routes, current_dist, removal_fraction):
        # identify routes with at least 1 customer (length > 2)
        valid_routes = [r for r in range(truck_count) if len(routes[r]) > 2]
        if len(valid_routes) < 2:
            return False  # cannot shake
        # sort by distance
        dist_with_idx = [(current_dist[r], r) for r in valid_routes]
        dist_with_idx.sort(key=lambda x: x[0])
        longest_route = dist_with_idx[-1][1]
        shortest_route = dist_with_idx[0][1]
        # remove customers
        removed = []
        # from longest: remove removal_fraction of its customers (at least 1)
        longest_cust_indices = list(range(1, len(routes[longest_route])-1))
        random.shuffle(longest_cust_indices)
        num_remove_long = max(1, int(len(longest_cust_indices) * removal_fraction * 1.5))
        num_remove_long = min(num_remove_long, len(longest_cust_indices))
        for idx in longest_cust_indices[:num_remove_long]:
            cust = routes[longest_route][idx]
            removed.append(cust)
        # sort indices in reverse to delete safely
        for idx in sorted(longest_cust_indices[:num_remove_long], reverse=True):
            del routes[longest_route][idx]
        # from shortest: remove removal_fraction of its customers
        shortest_cust_indices = list(range(1, len(routes[shortest_route])-1))
        random.shuffle(shortest_cust_indices)
        num_remove_short = max(1, int(len(shortest_cust_indices) * removal_fraction * 0.5))
        num_remove_short = min(num_remove_short, len(shortest_cust_indices))
        for idx in shortest_cust_indices[:num_remove_short]:
            cust = routes[shortest_route][idx]
            removed.append(cust)
        for idx in sorted(shortest_cust_indices[:num_remove_short], reverse=True):
            del routes[shortest_route][idx]
        # update current_dist for affected routes
        current_dist[longest_route] = route_distance(routes[longest_route])
        current_dist[shortest_route] = route_distance(routes[shortest_route])
        # reinsert removed customers using regret-2 with balancing
        lambda_balance = 0.5
        unassigned = removed[:]
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_route_idx = None
            best_pos = None
            best_cost_val = None
            for cust in unassigned:
                costs_info = []
                for r in range(truck_count):
                    base_cost, pos = insert_cost(routes[r], cust)
                    penalty = lambda_balance * (max(current_dist) - current_dist[r])
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
            max_dist = max(current_dist)
            avg_dist = sum(current_dist) / truck_count
            imbalance = max_dist - avg_dist
            lambda_balance = min(1.0, max(0.1, imbalance / max(avg_dist, 1e-9)))
        return True
    
    global_best_max = float('inf')
    global_best_routes = None
    num_restarts = 20
    no_improve_restarts = 0
    global_info = {'best_max': global_best_max, 'routes': None}
    
    for restart in range(num_restarts):
        customers = list(range(1, n))
        if restart >= 5 and no_improve_restarts >= 5:
            random.shuffle(customers)
        else:
            random.shuffle(customers)
        
        routes, current_dist = construct_routes(customers)
        best_max = max(current_dist)
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [list(r) for r in routes]
            global_info = {'best_max': global_best_max, 'routes': global_best_routes}
            report_best_vrp(routes)
            no_improve_restarts = 0
        else:
            no_improve_restarts += 1
        
        # local search and shake cycles
        shake_fraction = 0.3
        max_shakes = 5
        shake_counter = 0
        while shake_counter < max_shakes:
            best_max_before = best_max
            best_max, routes, current_dist = local_search(routes, current_dist, best_max, global_info)
            if best_max < best_max_before - 1e-12:
                # improvement, update global if needed
                if best_max < global_best_max:
                    global_best_max = best_max
                    global_best_routes = [list(r) for r in routes]
                    global_info = {'best_max': global_best_max, 'routes': global_best_routes}
                    report_best_vrp(routes)
                shake_counter = 0  # reset if improved
                shake_fraction = 0.3
            else:
                # no improvement, try shake
                if shake(routes, current_dist, shake_fraction):
                    shake_counter += 1
                    shake_fraction = max(0.1, shake_fraction - 0.05)
                    # after shake, re-evaluate
                    new_max = max(current_dist)
                    if new_max < best_max:
                        best_max = new_max
                        if best_max < global_best_max:
                            global_best_max = best_max
                            global_best_routes = [list(r) for r in routes]
                            global_info = {'best_max': global_best_max, 'routes': global_best_routes}
                            report_best_vrp(routes)
                else:
                    break
        # update global else from restart
        if global_best_routes is None:
            global_best_routes = [list(r) for r in routes]
            global_best_max = best_max
            global_info = {'best_max': global_best_max, 'routes': global_best_routes}
    
    return global_info['routes']
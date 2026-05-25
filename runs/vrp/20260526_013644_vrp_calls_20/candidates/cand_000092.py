import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    unassigned = set(customers)
    routes = [[depot, depot] for _ in range(truck_count)]
    
    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def best_insertion(cust, route):
        best_cost = float('inf')
        best_pos = -1
        for pos in range(1, len(route)):
            i = route[pos-1]
            j = route[pos]
            cost = distance_matrix[i, cust] + distance_matrix[cust, j] - distance_matrix[i, j]
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        return best_cost, best_pos
    
    # Regret-2 construction
    while unassigned:
        best_regret = -1
        best_cust = -1
        best_route_idx = -1
        best_pos = -1
        best_cost_for_cust = float('inf')
        for cust in unassigned:
            costs = []
            for r_idx, route in enumerate(routes):
                cost, pos = best_insertion(cust, route)
                costs.append((cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            if len(costs) == 1:
                regret = costs[0][0] * 2
            else:
                regret = costs[1][0] - costs[0][0]
            if regret > best_regret or (regret == best_regret and costs[0][0] > best_cost_for_cust):
                best_regret = regret
                best_cust = cust
                best_cost_for_cust = costs[0][0]
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
            elif regret == best_regret and costs[0][0] == best_cost_for_cust:
                if cust < best_cust:
                    best_cust = cust
                    best_route_idx = costs[0][1]
                    best_pos = costs[0][2]
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    n_cust = n - 1
    max_iters_per_restart = 5 * n_cust
    max_restarts = 3
    
    for restart in range(max_restarts):
        for _ in range(max_iters_per_restart):
            improved = False
            dists = [route_dist(r) for r in routes]
            # Inter-route relocate: consider moves from every route
            best_move = None
            best_new_max = best_max
            for src_idx in range(truck_count):
                src_route = routes[src_idx]
                if len(src_route) <= 2:
                    continue
                for cust_idx in range(1, len(src_route)-1):
                    cust = src_route[cust_idx]
                    new_src = src_route[:cust_idx] + src_route[cust_idx+1:]
                    for dst_idx in range(truck_count):
                        if dst_idx == src_idx:
                            continue
                        dst_route = routes[dst_idx]
                        cost, pos = best_insertion(cust, dst_route)
                        new_dst = list(dst_route)
                        new_dst.insert(pos, cust)
                        new_src_dist = route_dist(new_src)
                        new_dst_dist = route_dist(new_dst)
                        other_dists = [dists[i] for i in range(truck_count) if i != src_idx and i != dst_idx]
                        cand_max = max([new_src_dist, new_dst_dist] + other_dists)
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_move = (src_idx, dst_idx, new_src, new_dst)
            if best_move is not None:
                src_idx, dst_idx, new_src, new_dst = best_move
                routes[src_idx] = new_src
                routes[dst_idx] = new_dst
                best_max = best_new_max
                improved = True
                report_best_vrp(routes)
            else:
                # Intra-route 2-opt on longest route
                dists = [route_dist(r) for r in routes]
                max_idx = max(range(truck_count), key=lambda i: (dists[i], i))
                max_route = routes[max_idx]
                if len(max_route) > 3:
                    best_2opt = None
                    best_2opt_dist = route_dist(max_route)
                    for i in range(1, len(max_route)-2):
                        for j in range(i+1, len(max_route)-1):
                            new_route = max_route[:i] + max_route[i:j+1][::-1] + max_route[j+1:]
                            new_dist = route_dist(new_route)
                            if new_dist < best_2opt_dist:
                                best_2opt_dist = new_dist
                                best_2opt = (i, j, new_route)
                    if best_2opt is not None:
                        i, j, new_route = best_2opt
                        routes[max_idx] = new_route
                        # Update best_max
                        dists = [route_dist(r) for r in routes]
                        new_max = max(dists)
                        if new_max < best_max:
                            best_max = new_max
                            improved = True
                            report_best_vrp(routes)
            if not improved:
                break
        # Diversification restart: if not the last restart, perturb current routes
        if restart < max_restarts - 1:
            # Perform random swap of two customers from different routes (if possible)
            # Ensure at least two distinct routes have customers
            non_empty = [i for i, r in enumerate(routes) if len(r) > 2]
            if len(non_empty) >= 2:
                src_idx = random.choice(non_empty)
                dst_idx = random.choice([i for i in non_empty if i != src_idx])
                src_route = routes[src_idx]
                dst_route = routes[dst_idx]
                # pick random customer positions (not depot)
                src_pos = random.randint(1, len(src_route)-2)
                dst_pos = random.randint(1, len(dst_route)-2)
                # swap
                cust_src = src_route[src_pos]
                cust_dst = dst_route[dst_pos]
                new_src = list(src_route)
                new_dst = list(dst_route)
                new_src[src_pos] = cust_dst
                new_dst[dst_pos] = cust_src
                routes[src_idx] = new_src
                routes[dst_idx] = new_dst
                # Update best_max if needed (but best_max from previous run may still be lower)
                dists = [route_dist(r) for r in routes]
                new_max = max(dists)
                if new_max < best_max:
                    best_max = new_max
                    report_best_vrp(routes)
                else:
                    # revert? No, perturbation is intentional
                    pass
    # Ensure exactly truck_count routes, each [0,0] if empty
    result = []
    for r in routes:
        if len(r) <= 2:
            result.append([0,0])
        else:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
            result.append(r)
    while len(result) < truck_count:
        result.append([0,0])
    return result
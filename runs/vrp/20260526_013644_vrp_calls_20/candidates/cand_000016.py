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
    
    # Regret insertion construction
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
    max_iters = 10 * n_cust
    iter_count = 0
    improvement_occurred = True
    while iter_count < max_iters and improvement_occurred:
        improvement_occurred = False
        iter_count += 1
        # Compute current distances
        dists = [route_dist(r) for r in routes]
        current_max = max(dists)
        best_move = None
        best_new_max = current_max
        # Consider all routes as source
        for src_idx in range(truck_count):
            if len(routes[src_idx]) <= 2:
                continue
            src_route = routes[src_idx]
            for cust_pos in range(1, len(src_route)-1):
                cust = src_route[cust_pos]
                new_src = src_route[:cust_pos] + src_route[cust_pos+1:]
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    cost, pos = best_insertion(cust, dst_route)
                    new_dst = list(dst_route)
                    new_dst.insert(pos, cust)
                    new_src_dist = route_dist(new_src)
                    new_dst_dist = route_dist(new_dst)
                    new_dist_list = []
                    for i in range(truck_count):
                        if i == src_idx:
                            new_dist_list.append(new_src_dist)
                        elif i == dst_idx:
                            new_dist_list.append(new_dst_dist)
                        else:
                            new_dist_list.append(dists[i])
                    cand_max = max(new_dist_list)
                    if cand_max < best_new_max:
                        best_new_max = cand_max
                        best_move = (src_idx, dst_idx, cust_pos, pos, new_src, new_dst)
        if best_move is not None:
            src_idx, dst_idx, cust_pos, pos, new_src, new_dst = best_move
            routes[src_idx] = new_src
            routes[dst_idx] = new_dst
            improvement_occurred = True
            report_best_vrp(routes)
            dists = [route_dist(r) for r in routes]
            best_max = min(best_max, max(dists))
        else:
            # 2-opt on longest route
            dists = [route_dist(r) for r in routes]
            max_idx = max(range(truck_count), key=lambda i: (dists[i], -i))
            max_route = routes[max_idx]
            if len(max_route) <= 3:
                continue
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
                improvement_occurred = True
                report_best_vrp(routes)
                dists = [route_dist(r) for r in routes]
                best_max = min(best_max, max(dists))
            else:
                # Diversification: random relocate of up to 20% customers
                if len(customers) > 5:
                    num_relocate = max(1, int(0.2 * len(customers)))
                    all_cust = list(range(1, n))
                    random.shuffle(all_cust)
                    chosen = all_cust[:num_relocate]
                    for cust in chosen:
                        # find current position
                        for r_idx, route in enumerate(routes):
                            if cust in route:
                                src_idx = r_idx
                                cust_pos = route.index(cust)
                                break
                        # remove from source
                        new_src = routes[src_idx][:cust_pos] + routes[src_idx][cust_pos+1:]
                        routes[src_idx] = new_src
                        # reinsert randomly into another route (or same? better to different)
                        dst_idx = random.randint(0, truck_count-1)
                        dst_route = routes[dst_idx]
                        pos = random.randint(1, len(dst_route)-1)
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        routes[dst_idx] = new_dst
                    improvement_occurred = True  # force continue
    # Repair routes: ensure depot start/end
    for r in routes:
        if r[0] != 0:
            r.insert(0, 0)
        if r[-1] != 0:
            r.append(0)
    while len(routes) < truck_count:
        routes.append([0, 0])
    # Trim empty trucks
    result = [r if len(r) > 2 else [0,0] for r in routes]
    return result
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
    max_iters = 10 * n_cust  # bounded outer loop
    restart_num = 0
    max_restarts = 2 * n_cust
    for _ in range(max_iters):
        if restart_num > max_restarts:
            break
        improved = False
        # Compute current route distances
        dists = [route_dist(r) for r in routes]
        # Try relocations from all routes (broaden)
        best_move = None
        best_new_max = best_max
        for src_idx in range(truck_count):
            src_route = routes[src_idx]
            if len(src_route) <= 2:
                continue
            for cust_idx_in_route in range(1, len(src_route)-1):
                cust = src_route[cust_idx_in_route]
                new_src = src_route[:cust_idx_in_route] + src_route[cust_idx_in_route+1:]
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    cost, pos = best_insertion(cust, dst_route)
                    new_dst = list(dst_route)
                    new_dst.insert(pos, cust)
                    # Compute distances for affected routes
                    new_src_dist = route_dist(new_src)
                    new_dst_dist = route_dist(new_dst)
                    # Other distances unchanged
                    other_dists = [dists[i] for i in range(truck_count) if i != src_idx and i != dst_idx]
                    cand_max = max([new_src_dist, new_dst_dist] + other_dists)
                    if cand_max < best_new_max:
                        best_new_max = cand_max
                        best_move = (src_idx, dst_idx, cust_idx_in_route, pos, cust, new_src, new_dst)
        if best_move is not None:
            src_idx, dst_idx, cust_idx_in_route, pos, cust, new_src, new_dst = best_move
            routes[src_idx] = new_src
            routes[dst_idx] = new_dst
            best_max = best_new_max
            improved = True
            report_best_vrp(routes)
        else:
            # Intra-route 2-opt on longest route (tie break: smallest index)
            longest_idx = max(range(truck_count), key=lambda i: (dists[i], -i))
            longest_route = routes[longest_idx]
            if len(longest_route) > 3:
                best_2opt = None
                best_2opt_dist = route_dist(longest_route)
                for i in range(1, len(longest_route)-2):
                    for j in range(i+1, len(longest_route)-1):
                        new_route = longest_route[:i] + longest_route[i:j+1][::-1] + longest_route[j+1:]
                        new_dist = route_dist(new_route)
                        if new_dist < best_2opt_dist:
                            best_2opt_dist = new_dist
                            best_2opt = (i, j, new_route)
                if best_2opt is not None:
                    i, j, new_route = best_2opt
                    routes[longest_idx] = new_route
                    dists = [route_dist(r) for r in routes]
                    new_max = max(dists)
                    if new_max < best_max:
                        best_max = new_max
                        improved = True
                        report_best_vrp(routes)
        if not improved:
            # Diversification restart: randomly relocate a customer from longest route to a random other route
            longest_idx = max(range(truck_count), key=lambda i: (dists[i], -i))
            longest_route = routes[longest_idx]
            if len(longest_route) <= 2:
                break
            # Select a random customer from longest route (excluding depots)
            possible_pos = list(range(1, len(longest_route)-1))
            if not possible_pos:
                break
            rand_pos = random.choice(possible_pos)
            cust = longest_route[rand_pos]
            new_longest = longest_route[:rand_pos] + longest_route[rand_pos+1:]
            # Choose a random other route
            other_indices = [i for i in range(truck_count) if i != longest_idx]
            if not other_indices:
                break
            dst_idx = random.choice(other_indices)
            dst_route = routes[dst_idx]
            # Insert at a random position (excluding depot start/end? but can insert anywhere except at ends? Actually insertion can be between 1 and len-1)
            insert_pos = random.randint(1, len(dst_route)-1) if len(dst_route) > 1 else 1
            new_dst = list(dst_route)
            new_dst.insert(insert_pos, cust)
            routes[longest_idx] = new_longest
            routes[dst_idx] = new_dst
            dists = [route_dist(r) for r in routes]
            best_max = max(dists)
            restart_num += 1
            improved = True  # to continue loop
            report_best_vrp(routes)
    # Ensure exactly truck_count routes, each [0,0] if empty
    result = []
    for r in routes:
        if len(r) <= 2:
            result.append([0, 0])
        else:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
            result.append(r)
    while len(result) < truck_count:
        result.append([0, 0])
    return result
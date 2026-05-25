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
    
    # Regret-2 construction with epsilon-greedy
    epsilon = 0.2
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
        # epsilon-greedy: with probability epsilon, pick random customer
        if random.random() < epsilon:
            rand_cust = random.choice(list(unassigned))
            costs = []
            for r_idx, route in enumerate(routes):
                cost, pos = best_insertion(rand_cust, route)
                costs.append((cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            best_cust = rand_cust
            best_route_idx = costs[0][1]
            best_pos = costs[0][2]
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    n_cust = n - 1
    max_iters = 5 * n_cust
    stagnation = 0
    for _ in range(max_iters):
        improved = False
        dists = [route_dist(r) for r in routes]
        # Best inter-route relocate (all routes)
        best_move = None
        best_new_max = best_max
        for from_idx in range(truck_count):
            from_route = routes[from_idx]
            for cust_pos in range(1, len(from_route)-1):
                cust = from_route[cust_pos]
                new_from = from_route[:cust_pos] + from_route[cust_pos+1:]
                for to_idx in range(truck_count):
                    if to_idx == from_idx:
                        continue
                    cost, pos = best_insertion(cust, routes[to_idx])
                    new_to = list(routes[to_idx])
                    new_to.insert(pos, cust)
                    new_dists = [route_dist(r) if i == from_idx else route_dist(new_to) if i == to_idx else dists[i] for i in range(truck_count)]
                    cand_max = max(new_dists)
                    if cand_max < best_new_max:
                        best_new_max = cand_max
                        best_move = (from_idx, to_idx, cust_pos, pos, cust, new_from, new_to)
        if best_move is not None:
            from_idx, to_idx, cust_pos, pos, cust, new_from, new_to = best_move
            routes[from_idx] = new_from
            routes[to_idx] = new_to
            best_max = best_new_max
            improved = True
            stagnation = 0
            report_best_vrp([list(r) for r in routes])
        else:
            # Cross-exchange between longest route and others
            longest_idx = max(range(truck_count), key=lambda i: (dists[i], i))
            longest = routes[longest_idx]
            best_cross = None
            best_cross_max = best_max
            for other_idx in range(truck_count):
                if other_idx == longest_idx:
                    continue
                other = routes[other_idx]
                # Evaluate swaps of segments
                for i in range(1, len(longest)-1):
                    for j in range(1, len(other)-1):
                        # swap ending parts: from i to end with from j to end?
                        # Standard cross: exchange tails after i and j
                        new_long = longest[:i] + other[j:]
                        new_other = other[:j] + longest[i:]
                        # Ensure both start and end with depot
                        if new_long[-1] != 0:
                            new_long.append(0)
                        if new_other[-1] != 0:
                            new_other.append(0)
                        if new_long[0] != 0:
                            new_long.insert(0, 0)
                        if new_other[0] != 0:
                            new_other.insert(0, 0)
                        new_dists = [route_dist(new_long) if idx == longest_idx else route_dist(new_other) if idx == other_idx else dists[idx] for idx in range(truck_count)]
                        cand_max = max(new_dists)
                        if cand_max < best_cross_max:
                            best_cross_max = cand_max
                            best_cross = (longest_idx, other_idx, i, j, new_long, new_other)
            if best_cross is not None:
                longest_idx, other_idx, i, j, new_long, new_other = best_cross
                routes[longest_idx] = new_long
                routes[other_idx] = new_other
                best_max = best_cross_max
                improved = True
                stagnation = 0
                report_best_vrp([list(r) for r in routes])
            else:
                # Intra-route 2-opt on longest route
                if len(longest) > 3:
                    best_2opt = None
                    best_2opt_dist = route_dist(longest)
                    for i in range(1, len(longest)-2):
                        for j in range(i+1, len(longest)-1):
                            new_route = longest[:i] + longest[i:j+1][::-1] + longest[j+1:]
                            new_dist = route_dist(new_route)
                            if new_dist < best_2opt_dist:
                                best_2opt_dist = new_dist
                                best_2opt = (i, j, new_route)
                    if best_2opt is not None:
                        i, j, new_route = best_2opt
                        routes[longest_idx] = new_route
                        new_dists = [route_dist(r) for r in routes]
                        new_max = max(new_dists)
                        if new_max < best_max:
                            best_max = new_max
                            improved = True
                            stagnation = 0
                            report_best_vrp([list(r) for r in routes])
        if not improved:
            stagnation += 1
            if stagnation >= 40:
                # Perturbation: remove worst customers and reinsert
                # Determine customer contribution (cost of being in route)
                cust_contribution = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)-1):
                        cust = route[pos]
                        prev = route[pos-1]
                        nxt = route[pos+1]
                        contrib = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        cust_contribution.append((contrib, cust, r_idx, pos))
                # Sort descending by contribution
                cust_contribution.sort(reverse=True, key=lambda x: x[0])
                # Remove 20% of customers (or at least 1)
                num_remove = max(1, n_cust // 5)
                removed = []
                for _ in range(num_remove):
                    if not cust_contribution:
                        break
                    contrib, cust, r_idx, pos = cust_contribution.pop(0)
                    routes[r_idx].pop(pos)
                    removed.append(cust)
                # Reinsert removed customers using regret-2 (without epsilon to be deterministic)
                unassigned = set(removed)
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
                dists = [route_dist(r) for r in routes]
                new_max = max(dists)
                if new_max < best_max:
                    best_max = new_max
                    report_best_vrp([list(r) for r in routes])
                stagnation = 0
                improved = True  # to continue loop
        if not improved:
            if all(len(r) <= 2 for r in routes):
                break
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
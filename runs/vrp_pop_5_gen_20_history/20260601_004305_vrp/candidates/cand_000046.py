import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in range(1, n)]
        for _ in range(truck_count - m):
            routes.append([0, 0])
        return routes

    # Helper functions
    def route_cost(route):
        cost = 0.0
        for i in range(len(route)-1):
            cost += distance_matrix[route[i], route[i+1]]
        return cost

    def max_and_total(routes):
        maxd = 0.0
        totald = 0.0
        for r in routes:
            d = route_cost(r)
            totald += d
            if d > maxd:
                maxd = d
        return maxd, totald

    def best_insertion(customer, route):
        best_cost = math.inf
        best_pos = None
        # route has at least [0,0]
        for pos in range(1, len(route)):
            delta = (distance_matrix[route[pos-1], customer] +
                     distance_matrix[customer, route[pos]] -
                     distance_matrix[route[pos-1], route[pos]])
            if delta < best_cost:
                best_cost = delta
                best_pos = pos
        return best_pos, best_cost

    # Regret-2 construction
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    while unassigned:
        best_cust = None
        best_route_idx = None
        best_pos = None
        max_regret = -math.inf
        for cust in unassigned:
            costs = []
            positions = []
            for r_idx, route in enumerate(routes):
                pos, cost = best_insertion(cust, route)
                costs.append(cost)
                positions.append(pos)
            sorted_costs = sorted(costs)
            if len(sorted_costs) >= 2:
                regret = sorted_costs[1] - sorted_costs[0]
            else:
                regret = sorted_costs[0]
            if regret > max_regret:
                max_regret = regret
                best_cust = cust
                best_route_idx = int(np.argmin(costs))
                best_pos = positions[best_route_idx]
            elif regret == max_regret and (best_cust is None or cust < best_cust):
                best_cust = cust
                best_route_idx = int(np.argmin(costs))
                best_pos = positions[best_route_idx]
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)

    current_max, current_total = max_and_total(routes)
    best_routes = [list(r) for r in routes]
    best_max = current_max
    best_total = current_total
    report_best_vrp(best_routes)

    def two_opt(route):
        if len(route) <= 3:
            return route, False
        improved = False
        best_route = route[:]
        best_cost = route_cost(best_route)
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_cost = route_cost(new_route)
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
        return best_route, improved

    def relocate_move(routes):
        n_routes = len(routes)
        for src_idx in range(n_routes):
            src_route = routes[src_idx]
            for pos in range(1, len(src_route)-1):
                cust = src_route[pos]
                new_src = src_route[:pos] + src_route[pos+1:]
                src_cost = route_cost(new_src)
                for dst_idx in range(n_routes):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    for insert_pos in range(1, len(dst_route)):
                        new_dst = dst_route[:insert_pos] + [cust] + dst_route[insert_pos:]
                        dst_cost = route_cost(new_dst)
                        # compute new max and total
                        new_routes = routes.copy()
                        new_routes[src_idx] = new_src
                        new_routes[dst_idx] = new_dst
                        new_max, new_total = max_and_total(new_routes)
                        # if better than current best, apply immediately
                        if new_max < best_max or (new_max == best_max and new_total < best_total):
                            routes[src_idx] = new_src
                            routes[dst_idx] = new_dst
                            return True, new_max, new_total
        return False, None, None

    def swap_move(routes):
        n_routes = len(routes)
        for i_idx in range(n_routes):
            i_route = routes[i_idx]
            for pos_i in range(1, len(i_route)-1):
                cust_i = i_route[pos_i]
                for j_idx in range(n_routes):
                    if j_idx == i_idx:
                        continue
                    j_route = routes[j_idx]
                    for pos_j in range(1, len(j_route)-1):
                        cust_j = j_route[pos_j]
                        new_i = i_route[:pos_i] + [cust_j] + i_route[pos_i+1:]
                        new_j = j_route[:pos_j] + [cust_i] + j_route[pos_j+1:]
                        new_routes = routes.copy()
                        new_routes[i_idx] = new_i
                        new_routes[j_idx] = new_j
                        new_max, new_total = max_and_total(new_routes)
                        if new_max < best_max or (new_max == best_max and new_total < best_total):
                            routes[i_idx] = new_i
                            routes[j_idx] = new_j
                            return True, new_max, new_total
        return False, None, None

    def cross_exchange_move(routes):
        n_routes = len(routes)
        for i_idx in range(n_routes):
            i_route = routes[i_idx]
            for j_idx in range(i_idx+1, n_routes):
                j_route = routes[j_idx]
                # exchange segments of consecutive customers
                for i_start in range(1, len(i_route)-1):
                    for i_end in range(i_start, len(i_route)-1):
                        for j_start in range(1, len(j_route)-1):
                            for j_end in range(j_start, len(j_route)-1):
                                # new routes: swap segments
                                new_i = i_route[:i_start] + j_route[j_start:j_end+1] + i_route[i_end+1:]
                                new_j = j_route[:j_start] + i_route[i_start:i_end+1] + j_route[j_end+1:]
                                new_routes = routes.copy()
                                new_routes[i_idx] = new_i
                                new_routes[j_idx] = new_j
                                new_max, new_total = max_and_total(new_routes)
                                if new_max < best_max or (new_max == best_max and new_total < best_total):
                                    routes[i_idx] = new_i
                                    routes[j_idx] = new_j
                                    return True, new_max, new_total
        return False, None, None

    def local_search(routes):
        improved = True
        while improved:
            improved = False
            # intra 2-opt
            for idx in range(len(routes)):
                new_route, imp = two_opt(routes[idx])
                if imp:
                    routes[idx] = new_route
                    new_max, new_total = max_and_total(routes)
                    if new_max < best_max or (new_max == best_max and new_total < best_total):
                        best_max, best_total = new_max, new_total
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                    improved = True
            # inter relocate
            imp, new_max, new_total = relocate_move(routes)
            if imp:
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max, best_total = new_max, new_total
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                improved = True
            # inter swap
            imp, new_max, new_total = swap_move(routes)
            if imp:
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max, best_total = new_max, new_total
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                improved = True
            # cross exchange
            imp, new_max, new_total = cross_exchange_move(routes)
            if imp:
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max, best_total = new_max, new_total
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                improved = True
        return routes

    # Apply local search once
    routes = local_search(routes)

    # VNS shaking and local search
    max_vns_iter = n * 2
    neighborhoods = ['relocate', 'swap', 'cross']
    for _ in range(max_vns_iter):
        # Shake: randomly pick a neighborhood and apply random move
        shaken = False
        attempts = 0
        while not shaken and attempts < 100:
            attempts += 1
            nh = random.choice(neighborhoods)
            if nh == 'relocate':
                # random relocate
                src_idx = random.randrange(truck_count)
                if len(routes[src_idx]) <= 3:
                    continue
                pos = random.randint(1, len(routes[src_idx])-2)
                cust = routes[src_idx].pop(pos)
                dst_idx = random.randrange(truck_count)
                insert_pos = random.randint(1, len(routes[dst_idx])-1)
                routes[dst_idx].insert(insert_pos, cust)
                shaken = True
            elif nh == 'swap':
                i = random.randrange(truck_count)
                j = random.randrange(truck_count)
                if i == j or len(routes[i]) <= 3 or len(routes[j]) <= 3:
                    continue
                pos_i = random.randint(1, len(routes[i])-2)
                pos_j = random.randint(1, len(routes[j])-2)
                cust_i = routes[i][pos_i]
                cust_j = routes[j][pos_j]
                routes[i][pos_i] = cust_j
                routes[j][pos_j] = cust_i
                shaken = True
            elif nh == 'cross':
                i = random.randrange(truck_count)
                j = random.randrange(truck_count)
                if i == j or len(routes[i]) <= 3 or len(routes[j]) <= 3:
                    continue
                i_start = random.randint(1, len(routes[i])-2)
                i_end = random.randint(i_start, len(routes[i])-2)
                j_start = random.randint(1, len(routes[j])-2)
                j_end = random.randint(j_start, len(routes[j])-2)
                seg_i = routes[i][i_start:i_end+1]
                seg_j = routes[j][j_start:j_end+1]
                routes[i] = routes[i][:i_start] + seg_j + routes[i][i_end+1:]
                routes[j] = routes[j][:j_start] + seg_i + routes[j][j_end+1:]
                shaken = True
        if not shaken:
            break
        # Local search after shaking
        routes = local_search(routes)
        new_max, new_total = max_and_total(routes)
        if new_max < best_max or (new_max == best_max and new_total < best_total):
            best_max, best_total = new_max, new_total
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        else:
            # revert to best? no, accept if equal? for VNS standard, we accept if equal or worse? but we want exploration
            pass

    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes
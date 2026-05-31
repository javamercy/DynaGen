import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def regret_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= 3:
                    regret = incs[1][0] - incs[0][0] + incs[2][0] - incs[0][0]
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route = r_idx
                    best_pos = pos
            routes[best_route].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    def evaluate(routes):
        return max(route_distance(r) for r in routes)

    def vnd(routes):
        lengths = [route_distance(r) for r in routes]
        improved = True
        while improved:
            improved = False
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            # relocate
            for cust in range(1, n):
                src_idx = None
                src_pos = None
                for i, route in enumerate(routes):
                    if cust in route:
                        src_idx = i
                        src_pos = route.index(cust)
                        break
                if src_idx is None:
                    continue
                new_src = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
                src_len = route_distance(new_src)
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    if len(routes[dst_idx]) <= 2:
                        continue
                    for ins_pos in range(1, len(routes[dst_idx])):
                        new_dst = routes[dst_idx][:ins_pos] + [cust] + routes[dst_idx][ins_pos:]
                        new_lengths = lengths[:]
                        new_lengths[src_idx] = src_len
                        new_lengths[dst_idx] = route_distance(new_dst)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('relocate', src_idx, src_pos, dst_idx, ins_pos, new_src, new_dst)
            # swap
            for i in range(truck_count):
                if len(routes[i]) <= 2:
                    continue
                for ip in range(1, len(routes[i])-1):
                    cust_i = routes[i][ip]
                    for j in range(i+1, truck_count):
                        if len(routes[j]) <= 2:
                            continue
                        for jp in range(1, len(routes[j])-1):
                            cust_j = routes[j][jp]
                            new_i = routes[i][:ip] + [cust_j] + routes[i][ip+1:]
                            new_j = routes[j][:jp] + [cust_i] + routes[j][jp+1:]
                            new_lengths = lengths[:]
                            new_lengths[i] = route_distance(new_i)
                            new_lengths[j] = route_distance(new_j)
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                                best_new_max = new_max
                                best_total = new_total
                                best_move = ('swap', i, ip, j, jp, new_i, new_j)
            # 2-opt
            for i in range(truck_count):
                if len(routes[i]) <= 3:
                    continue
                for a in range(1, len(routes[i])-2):
                    for b in range(a+1, len(routes[i])-1):
                        new_route = routes[i][:a] + routes[i][a:b+1][::-1] + routes[i][b+1:]
                        new_len = route_distance(new_route)
                        if new_len >= lengths[i]:
                            continue
                        new_lengths = lengths[:]
                        new_lengths[i] = new_len
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('2opt', i, a, b, new_route)
            if best_move is not None and best_new_max < max(lengths):
                if best_move[0] == 'relocate':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                elif best_move[0] == 'swap':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                else:
                    routes[best_move[1]] = best_move[4]
                lengths = [route_distance(r) for r in routes]
                improved = True
        return routes, lengths

    # Build initial solution
    best_routes = regret_construction()
    best_routes, best_lengths = vnd(best_routes)
    best_max = max(best_lengths)
    report_best_vrp(best_routes)

    # ILS iterations
    ils_iters = min(20, n // 2)
    for _ in range(ils_iters):
        # Shake: random relocate of 1 to 3 customers
        routes_shake = [route[:] for route in best_routes]
        k = random.randint(1, 3)
        for _ in range(k):
            # pick a random customer from a non-empty route
            non_empty = [i for i, r in enumerate(routes_shake) if len(r) > 2]
            if not non_empty:
                break
            src_idx = random.choice(non_empty)
            route = routes_shake[src_idx]
            pos = random.randint(1, len(route)-2)
            cust = route[pos]
            new_src = route[:pos] + route[pos+1:]
            # pick destination route and insertion position
            dst_idx = random.randrange(truck_count)
            dst_route = routes_shake[dst_idx]
            ins_pos = random.randint(1, len(dst_route)-1) if len(dst_route) > 2 else 1
            new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
            routes_shake[src_idx] = new_src
            routes_shake[dst_idx] = new_dst
        # Apply VND
        routes_shake, lengths_shake = vnd(routes_shake)
        new_max = max(lengths_shake)
        if new_max < best_max:
            best_max = new_max
            best_routes = routes_shake
            report_best_vrp(best_routes)

    return best_routes
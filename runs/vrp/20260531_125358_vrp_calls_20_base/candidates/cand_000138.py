import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(route, max_iter=5):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            best_i, best_j = None, None
            best_dist = route_distance(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-12:
                        best_dist = new_dist
                        best_i, best_j = i, j
            if best_i is not None:
                route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                improved = True
        return route

    def balance_routes(routes, lengths):
        max_balance_iter = n
        for _ in range(max_balance_iter):
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] - lengths[min_idx] < 1e-12:
                break
            max_route = routes[max_idx]
            best_reduction = 0
            best_cust_pos = None
            best_insert_route = None
            best_insert_pos = None
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                for dst_idx in range(truck_count):
                    if dst_idx == max_idx:
                        continue
                    dst_route = routes[dst_idx]
                    for p in range(1, len(dst_route)):
                        new_dst_route = dst_route[:p] + [cust] + dst_route[p:]
                        new_dst_len = route_distance(new_dst_route)
                        new_lengths = lengths[:]
                        new_lengths[max_idx] = new_max_len
                        new_lengths[dst_idx] = new_dst_len
                        new_max_global = max(new_lengths)
                        reduction = lengths[max_idx] - new_max_global
                        if reduction > best_reduction + 1e-12:
                            best_reduction = reduction
                            best_cust_pos = (max_idx, pos)
                            best_insert_route = dst_idx
                            best_insert_pos = p
            if best_reduction > 1e-12:
                max_idx, pos = best_cust_pos
                cust = routes[max_idx][pos]
                routes[max_idx] = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                routes[best_insert_route] = routes[best_insert_route][:best_insert_pos] + [cust] + routes[best_insert_route][best_insert_pos:]
                lengths = [route_distance(r) for r in routes]
            else:
                break
        return routes, lengths

    def regret_construction(k=3):
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: (x[0], x[2], x[1]))
                if len(incs) >= k:
                    regret = sum(incs[i][0] - incs[0][0] for i in range(1, k))
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret + 1e-12 or (abs(regret - best_regret) < 1e-12 and inc < best_inc - 1e-12):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    def ruin_recreate(routes, lengths, fraction=0.2, k=3):
        n_cust = n - 1
        num_remove = max(1, int(n_cust * fraction))
        removed = random.sample(customers, num_remove)
        new_routes = [[0, 0] for _ in range(truck_count)]
        for r_idx, route in enumerate(routes):
            new_routes[r_idx] = [0] + [c for c in route[1:-1] if c not in removed] + [0]
        unvisited = set(removed)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: (x[0], x[2], x[1]))
                if len(incs) >= k:
                    regret = sum(incs[i][0] - incs[0][0] for i in range(1, k))
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret + 1e-12 or (abs(regret - best_regret) < 1e-12 and inc < best_inc - 1e-12):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            new_routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        for r_idx in range(truck_count):
            if len(new_routes[r_idx]) > 2:
                new_routes[r_idx] = two_opt(new_routes[r_idx], max_iter=5)
        new_lengths = [route_distance(r) for r in new_routes]
        new_routes, new_lengths = balance_routes(new_routes, new_lengths)
        return new_routes, new_lengths

    best_routes = None
    best_max = float('inf')
    best_total = float('inf')
    num_restarts = 3
    for restart in range(num_restarts):
        routes = regret_construction(k=3)
        lengths = [route_distance(r) for r in routes]
        routes, lengths = balance_routes(routes, lengths)
        current_max = max(lengths)
        current_total = sum(lengths)
        if current_max < best_max - 1e-12 or (abs(current_max - best_max) < 1e-12 and current_total < best_total - 1e-12):
            best_max = current_max
            best_total = current_total
            best_routes = [r[:] for r in routes]
            # report_best_vrp(best_routes)  # callback removed for self-contained, but assumption is it's defined

        max_iter = min(n * truck_count, 100)  # bounded to avoid timeout
        for iteration in range(max_iter):
            applied_moves = False
            # 2-opt intra-route: best improvement
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                best_route = route
                best_len = route_distance(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = route_distance(new_route)
                        if new_len < best_len - 1e-12:
                            best_len = new_len
                            best_route = new_route
                if best_len < route_distance(route) - 1e-12:
                    routes[r_idx] = best_route
                    applied_moves = True
            # Inter-route swap: best improvement
            best_move = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            for i_idx in range(truck_count):
                for i_pos in range(1, len(routes[i_idx])-1):
                    cust_i = routes[i_idx][i_pos]
                    for j_idx in range(i_idx+1, truck_count):
                        for j_pos in range(1, len(routes[j_idx])-1):
                            cust_j = routes[j_idx][j_pos]
                            new_i = routes[i_idx][:i_pos] + [cust_j] + routes[i_idx][i_pos+1:]
                            new_j = routes[j_idx][:j_pos] + [cust_i] + routes[j_idx][j_pos+1:]
                            new_lengths = lengths[:]
                            new_lengths[i_idx] = route_distance(new_i)
                            new_lengths[j_idx] = route_distance(new_j)
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if new_max < best_new_max - 1e-12 or (abs(new_max - best_new_max) < 1e-12 and new_total < best_new_total - 1e-12):
                                best_new_max = new_max
                                best_new_total = new_total
                                best_move = (i_idx, i_pos, j_idx, j_pos, new_i, new_j)
            if best_move is not None:
                i_idx, i_pos, j_idx, j_pos, new_i, new_j = best_move
                routes[i_idx] = new_i
                routes[j_idx] = new_j
                lengths = [route_distance(r) for r in routes]
                applied_moves = True
            if not applied_moves:
                # Perturbation
                routes, lengths = ruin_recreate(routes, lengths, fraction=0.2, k=3)
            else:
                routes, lengths = balance_routes(routes, lengths)
            current_max = max(lengths)
            current_total = sum(lengths)
            if current_max < best_max - 1e-12 or (abs(current_max - best_max) < 1e-12 and current_total < best_total - 1e-12):
                best_max = current_max
                best_total = current_total
                best_routes = [r[:] for r in routes]
                # report_best_vrp(best_routes)
        # after improvement loop, do one more perturbation
        routes, lengths = ruin_recreate(routes, lengths, fraction=0.2, k=3)
        current_max = max(lengths)
        current_total = sum(lengths)
        if current_max < best_max - 1e-12 or (abs(current_max - best_max) < 1e-12 and current_total < best_total - 1e-12):
            best_max = current_max
            best_total = current_total
            best_routes = [r[:] for r in routes]
            # report_best_vrp(best_routes)

    # Fallback: if still no solution (shouldn't happen), build a simple nearest neighbor
    if best_routes is None:
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            for r_idx in range(truck_count):
                if not unvisited:
                    break
                last = routes[r_idx][-2] if len(routes[r_idx]) > 1 else 0
                best_cust = None
                best_dist = float('inf')
                for cust in unvisited:
                    d = distance_matrix[last, cust]
                    if d < best_dist:
                        best_dist = d
                        best_cust = cust
                if best_cust is not None:
                    routes[r_idx].insert(-1, best_cust)
                    unvisited.remove(best_cust)
        best_routes = routes

    return best_routes
import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

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
                incs.sort(key=lambda x: x[0])
                if len(incs) >= k:
                    regret = sum(incs[i][0] - incs[0][0] for i in range(1, k))
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    def improve_routes(routes):
        lengths = [route_distance(r) for r in routes]
        current_max = max(lengths)
        best_max = current_max
        best_total = sum(lengths)
        # VND loop limited by instance size
        for _ in range(n):
            improved = False
            # Intra-route 2-opt for each route
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                local_improved = True
                while local_improved:
                    local_improved = False
                    best_new_len = lengths[r_idx]
                    best_i, best_j = -1, -1
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_len = route_distance(new_route)
                            if new_len < best_new_len:
                                best_new_len = new_len
                                best_i, best_j = i, j
                    if best_i != -1:
                        route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                        routes[r_idx] = route
                        lengths[r_idx] = best_new_len
                        local_improved = True
                        improved = True
            # Inter-route relocate (best improvement)
            best_move = None
            best_new_max = current_max
            best_new_total = sum(lengths)
            for cust in range(1, n):
                src_idx = None
                src_pos = None
                for r_idx, route in enumerate(routes):
                    if cust in route:
                        src_idx = r_idx
                        src_pos = route.index(cust)
                        break
                if src_idx is None:
                    continue
                new_src_route = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
                src_len = route_distance(new_src_route)
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    for ins_pos in range(1, len(dst_route)):
                        new_dst_route = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                        new_lengths = lengths[:]
                        new_lengths[src_idx] = src_len
                        new_lengths[dst_idx] = route_distance(new_dst_route)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if (new_max < best_new_max) or (new_max == best_new_max and new_total < best_new_total):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_move = ('relocate', src_idx, src_pos, dst_idx, ins_pos, new_src_route, new_dst_route)
            if best_move is not None and best_new_max < current_max:
                routes[best_move[1]] = best_move[5]
                routes[best_move[3]] = best_move[6]
                lengths[best_move[1]] = route_distance(best_move[5])
                lengths[best_move[3]] = route_distance(best_move[6])
                current_max = best_new_max
                improved = True
            # Inter-route swap (best improvement)
            best_move = None
            best_new_max = current_max
            best_new_total = sum(lengths)
            for i_idx in range(truck_count):
                i_route = routes[i_idx]
                if len(i_route) <= 2:
                    continue
                for i_pos in range(1, len(i_route)-1):
                    cust_i = i_route[i_pos]
                    for j_idx in range(i_idx+1, truck_count):
                        j_route = routes[j_idx]
                        if len(j_route) <= 2:
                            continue
                        for j_pos in range(1, len(j_route)-1):
                            cust_j = j_route[j_pos]
                            new_i_route = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                            new_j_route = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                            new_lengths = lengths[:]
                            new_lengths[i_idx] = route_distance(new_i_route)
                            new_lengths[j_idx] = route_distance(new_j_route)
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if (new_max < best_new_max) or (new_max == best_new_max and new_total < best_new_total):
                                best_new_max = new_max
                                best_new_total = new_total
                                best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i_route, new_j_route)
            if best_move is not None and best_new_max < current_max:
                routes[best_move[1]] = best_move[5]
                routes[best_move[3]] = best_move[6]
                lengths[best_move[1]] = route_distance(best_move[5])
                lengths[best_move[3]] = route_distance(best_move[6])
                current_max = best_new_max
                improved = True
            if not improved:
                break
        return routes, lengths

    routes = regret_construction(k=3)
    lengths = [route_distance(r) for r in routes]
    best_max = max(lengths)
    best_routes = [r[:] for r in routes]
    report_best_vrp(best_routes)
    routes, lengths = improve_routes(routes)
    current_max = max(lengths)
    if current_max < best_max:
        best_routes = [r[:] for r in routes]
        report_best_vrp(best_routes)
    return best_routes
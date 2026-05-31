import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        if len(route) <= 2:
            return 0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def regret_insertion_construction():
        routes = [[depot, depot] for _ in range(truck_count)]
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
                        inc = (distance_matrix[route[pos-1], cust] +
                               distance_matrix[cust, route[pos]] -
                               distance_matrix[route[pos-1], route[pos]])
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= 2:
                    regret = incs[1][0] - incs[0][0]
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if (regret > best_regret or
                    (regret == best_regret and inc < best_inc)):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    best_routes = None
    best_max = float('inf')
    num_restarts = max(1, min(5, n // 10))
    for restart in range(num_restarts):
        routes = regret_insertion_construction()
        lengths = [route_distance(r) for r in routes]
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        improved = True
        while improved:
            improved = False
            best_move = None
            best_new_max = current_max
            best_new_total = sum(lengths)

            # Inter-route relocate
            for src_idx in range(truck_count):
                src_route = routes[src_idx]
                if len(src_route) <= 2:
                    continue
                for pos in range(1, len(src_route)-1):
                    cust = src_route[pos]
                    new_src = src_route[:pos] + src_route[pos+1:]
                    new_src_len = route_distance(new_src)
                    for dst_idx in range(truck_count):
                        if dst_idx == src_idx:
                            continue
                        dst_route = routes[dst_idx]
                        for ins_pos in range(1, len(dst_route)):
                            new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                            new_dst_len = route_distance(new_dst)
                            new_lengths = lengths[:]
                            new_lengths[src_idx] = new_src_len
                            new_lengths[dst_idx] = new_dst_len
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if (new_max < best_new_max or
                                (new_max == best_new_max and new_total < best_new_total) or
                                (new_max == best_new_max and new_total == best_new_total and src_idx < best_move[0] if best_move else True)):
                                best_new_max = new_max
                                best_new_total = new_total
                                best_move = ('relocate', src_idx, pos, dst_idx, ins_pos, new_src, new_dst)

            # Inter-route swap
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
                            new_i = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                            new_j = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                            new_i_len = route_distance(new_i)
                            new_j_len = route_distance(new_j)
                            new_lengths = lengths[:]
                            new_lengths[i_idx] = new_i_len
                            new_lengths[j_idx] = new_j_len
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if (new_max < best_new_max or
                                (new_max == best_new_max and new_total < best_new_total) or
                                (new_max == best_new_max and new_total == best_new_total and i_idx < j_idx)):
                                best_new_max = new_max
                                best_new_total = new_total
                                best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i, new_j)

            if best_move is not None and best_new_max < current_max:
                if best_move[0] == 'relocate':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                else:  # swap
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                lengths = [route_distance(r) for r in routes]
                current_max = max(lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                improved = True
    return best_routes
import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def cheapest_insertion():
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        if inc < best_inc:
                            best_inc = inc
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    def ruin_recreate(routes, fraction=0.15):
        n_cust = n - 1
        num_remove = max(1, int(n_cust * fraction))
        custs = list(range(1, n))
        random.shuffle(custs)
        to_remove = custs[:num_remove]
        new_routes = [[0, 0] for _ in range(truck_count)]
        for r_idx, route in enumerate(routes):
            new_routes[r_idx] = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
        unvisited = set(to_remove)
        while unvisited:
            best_cust = None
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                for r_idx, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        if inc < best_inc:
                            best_inc = inc
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
            new_routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return new_routes

    routes = cheapest_insertion()
    lengths = [route_distance(r) for r in routes]
    best_max = max(lengths)
    best_routes = [r[:] for r in routes]
    report_best_vrp(best_routes)

    max_iter = n * truck_count
    for _ in range(max_iter):
        best_move = None
        best_new_max = max(lengths)
        best_total = sum(lengths)

        # Inter-route relocate
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
                if len(dst_route) <= 2:
                    continue
                for ins_pos in range(1, len(dst_route)):
                    new_dst_route = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                    new_lengths = lengths[:]
                    new_lengths[src_idx] = src_len
                    new_lengths[dst_idx] = route_distance(new_dst_route)
                    new_max = max(new_lengths)
                    new_total = sum(new_lengths)
                    if (new_max < best_new_max) or (new_max == best_new_max and new_total < best_total):
                        best_new_max = new_max
                        best_total = new_total
                        best_move = ('relocate', cust, src_idx, dst_idx, ins_pos, new_src_route, new_dst_route)

        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = route_distance(new_route)
                    if new_len >= lengths[r_idx]:
                        continue
                    new_lengths = lengths[:]
                    new_lengths[r_idx] = new_len
                    new_max = max(new_lengths)
                    new_total = sum(new_lengths)
                    if (new_max < best_new_max) or (new_max == best_new_max and new_total < best_total):
                        best_new_max = new_max
                        best_total = new_total
                        best_move = ('2opt', r_idx, new_route)

        if best_move is not None and best_new_max < max(lengths):
            if best_move[0] == 'relocate':
                routes[best_move[2]] = best_move[5]
                routes[best_move[3]] = best_move[6]
            else:
                routes[best_move[1]] = best_move[2]
            lengths = [route_distance(r) for r in routes]
            current_max = max(lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
        else:
            routes = ruin_recreate(routes)
            lengths = [route_distance(r) for r in routes]
            current_max = max(lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)

    return best_routes
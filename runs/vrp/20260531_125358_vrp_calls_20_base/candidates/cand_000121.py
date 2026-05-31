import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(route, max_iter=10):
        best = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_distance(new_route) < route_distance(best):
                        best = new_route
                        improved = True
        return best

    def or_opt(route, max_iter=5):
        best = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    for segment_length in range(1, min(3, len(best)-j)):
                        segment = best[i:i+segment_length]
                        new_route = best[:i] + best[i+segment_length:j+1] + segment + best[j+1:]
                        if route_distance(new_route) < route_distance(best):
                            best = new_route
                            improved = True
        return best

    def vnd(routes, lengths, max_iters=5):
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            # Inter-route relocate (best improvement)
            best_move = None
            best_max = max(lengths)
            best_total = sum(lengths)
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
                        if new_max < best_max or (new_max == best_max and new_total < best_total):
                            best_max = new_max
                            best_total = new_total
                            best_move = ('relocate', cust, src_idx, src_pos, dst_idx, ins_pos, new_src_route, new_dst_route)
            if best_move is not None and best_max < max(lengths):
                _, cust, src_idx, src_pos, dst_idx, ins_pos, new_src, new_dst = best_move
                routes[src_idx] = new_src
                routes[dst_idx] = new_dst
                lengths = [route_distance(r) for r in routes]
                improved = True
            # Inter-route swap
            best_move = None
            current_max = max(lengths)
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
                            if new_max < current_max or (new_max == current_max and new_total < sum(lengths)):
                                current_max = new_max
                                best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i_route, new_j_route)
            if best_move is not None and best_move[0] != '':
                _, i_idx, i_pos, j_idx, j_pos, new_i, new_j = best_move
                routes[i_idx] = new_i
                routes[j_idx] = new_j
                lengths = [route_distance(r) for r in routes]
                improved = True
            # Intra-route 2-opt (best improvement on each route)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                best_route = two_opt(route, max_iter=5)
                if route_distance(best_route) < route_distance(route):
                    routes[r_idx] = best_route
                    lengths[r_idx] = route_distance(best_route)
                    improved = True
            # Intra-route Or-opt (on each route)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                best_route = or_opt(route, max_iter=2)
                if route_distance(best_route) < route_distance(route):
                    routes[r_idx] = best_route
                    lengths[r_idx] = route_distance(best_route)
                    improved = True
        return routes, lengths

    def balance(routes, lengths):
        # Move customers from longest to shortest until no improvement in max
        improved = True
        it = 0
        max_iter_balance = n
        while improved and it < max_iter_balance:
            improved = False
            it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_reduction = 0
            best_pos = None
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_insert_len = float('inf')
                best_insert_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_insert_len:
                        best_insert_len = l
                        best_insert_pos = p
                new_min_len = best_insert_len
                other_max = max([lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)] + [0])
                new_global_max = max(new_max_len, new_min_len, other_max)
                reduction = max(lengths) - new_global_max
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cust = cust
                    best_pos = best_insert_pos
            if best_cust is not None and best_reduction > 0:
                # Remove from max route
                new_max = [node for node in max_route if node != best_cust]
                # Insert into min route
                min_route = routes[min_idx]
                new_min = min_route[:best_pos] + [best_cust] + min_route[best_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_distance(new_max)
                lengths[min_idx] = route_distance(new_min)
                improved = True
        return routes, lengths

    def regret_construction(k=5):
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
        lengths = [route_distance(r) for r in routes]
        return routes, lengths

    def perturb(routes, lengths, fraction=0.1):
        # Remove customers from the longest route
        max_idx = max(range(truck_count), key=lambda i: lengths[i])
        route = routes[max_idx]
        num_remove = max(1, int((len(route)-2) * fraction))
        if num_remove >= len(route)-2:
            num_remove = len(route)-2
        # remove random customers from longest route
        remove_candidates = route[1:-1].copy()
        random.shuffle(remove_candidates)
        to_remove = set(remove_candidates[:num_remove])
        new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
        new_routes = routes[:]
        new_routes[max_idx] = new_route
        new_lengths = [route_distance(r) for r in new_routes]
        unvisited = to_remove
        # Reinsert with regret
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
                incs.sort(key=lambda x: x[0])
                if len(incs) >= 5:
                    regret = sum(incs[i][0] - incs[0][0] for i in range(1, 5))
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
            new_routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        new_lengths = [route_distance(r) for r in new_routes]
        new_routes, new_lengths = vnd(new_routes, new_lengths, max_iters=3)
        new_routes, new_lengths = balance(new_routes, new_lengths)
        return new_routes, new_lengths

    # Construction
    routes, lengths = regret_construction(k=5)
    routes, lengths = vnd(routes, lengths, max_iters=10)
    routes, lengths = balance(routes, lengths)
    best_routes = [r[:] for r in routes]
    best_max = max(lengths)
    report_best_vrp(best_routes)

    # Intensify loop
    for restart in range(3):  # bounded restarts
        for iteration in range(n * truck_count):
            # Perturb with small fraction on longest route
            new_routes, new_lengths = perturb(routes, lengths, fraction=0.1)
            new_max = max(new_lengths)
            if new_max < best_max:
                best_max = new_max
                best_routes = [r[:] for r in new_routes]
                report_best_vrp(best_routes)
            # Accept if max not worse too much
            if new_max <= max(lengths) * 1.05:  # allow slight worsening
                routes = new_routes
                lengths = new_lengths
            else:
                # Revert to best
                routes = [r[:] for r in best_routes]
                lengths = [route_distance(r) for r in routes]
    return best_routes
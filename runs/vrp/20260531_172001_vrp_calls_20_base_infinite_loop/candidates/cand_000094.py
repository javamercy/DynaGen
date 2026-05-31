import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def max_route_len(routes):
        return max(route_length(r) for r in routes)

    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 10)

    for _ in range(max_attempts):
        # Construction: min-max greedy with regret-2 tie-breaking
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            candidates = []
            for cust in unassigned:
                insert_info = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_info.append((new_max, cost, r_idx, pos))
                insert_info.sort(key=lambda x: (x[0], x[1]))
                best = insert_info[0]
                second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            chosen = candidates[0]
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)

        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

        # Improvement phase
        stagnation = 0
        max_iter = n * truck_count * 2
        iter_count = 0
        perturbation_size = max(1, n // 10)
        perturbation_threshold = 5

        while iter_count < max_iter:
            improved_this_iter = False

            # VND with neighborhoods: inter_relocate, inter_swap, intra_2opt, intra_or_opt
            neighborhoods = ['inter_relocate', 'inter_swap', 'intra_2opt', 'intra_or_opt']
            for nh in neighborhoods:
                if improved_this_iter:
                    break

                if nh == 'inter_relocate':
                    lengths = [route_length(r) for r in routes]
                    max_idx = int(np.argmax(lengths))
                    max_route = routes[max_idx]
                    if len(max_route) > 2:
                        best_delta = 0.0
                        best_move = None
                        for cust in max_route[1:-1]:
                            new_max_route = [x for x in max_route if x != cust]
                            new_max_len = route_length(new_max_route)
                            for r_idx in range(truck_count):
                                if r_idx == max_idx:
                                    continue
                                other_route = routes[r_idx]
                                for pos in range(1, len(other_route)):
                                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                                    new_other_len = route_length(new_other)
                                    other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                                    new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                                    if new_max_candidate < current_max - 1e-12:
                                        delta = current_max - new_max_candidate
                                        if delta > best_delta:
                                            best_delta = delta
                                            best_move = (cust, max_idx, r_idx, pos)
                        if best_move is not None:
                            cust, from_idx, to_idx, pos = best_move
                            routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                            routes[to_idx].insert(pos, cust)
                            current_max = current_max - best_delta
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                            # Intensify: immediate 2-opt on affected route
                            for r in [from_idx, to_idx]:
                                route = routes[r]
                                if len(route) > 3:
                                    imp = True
                                    while imp:
                                        imp = False
                                        for i in range(1, len(route)-2):
                                            for k in range(i+1, len(route)-1):
                                                new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                                                if route_length(new_route) < route_length(route) - 1e-12:
                                                    route = new_route
                                                    imp = True
                                                    break
                                            if imp:
                                                break
                                    routes[r] = route
                            current_max = max_route_len(routes)
                            improved_this_iter = True

                elif nh == 'inter_swap':
                    lengths = [route_length(r) for r in routes]
                    max_idx = int(np.argmax(lengths))
                    max_route = routes[max_idx]
                    if len(max_route) > 2:
                        best_delta = 0.0
                        best_move = None
                        for cust_i in max_route[1:-1]:
                            for other_idx in range(truck_count):
                                if other_idx == max_idx:
                                    continue
                                other_route = routes[other_idx]
                                for cust_j in other_route[1:-1]:
                                    new_max_route = [x if x != cust_i else cust_j for x in max_route]
                                    new_other_route = [x if x != cust_j else cust_i for x in other_route]
                                    new_max_len = route_length(new_max_route)
                                    new_other_len = route_length(new_other_route)
                                    other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, other_idx)]
                                    new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                                    if new_max_candidate < current_max - 1e-12:
                                        delta = current_max - new_max_candidate
                                        if delta > best_delta:
                                            best_delta = delta
                                            best_move = (cust_i, max_idx, cust_j, other_idx)
                        if best_move is not None:
                            cust_i, from_idx, cust_j, to_idx = best_move
                            routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                            routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                            current_max = current_max - best_delta
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                            # Intensify: immediate 2-opt on affected routes
                            for r in [from_idx, to_idx]:
                                route = routes[r]
                                if len(route) > 3:
                                    imp = True
                                    while imp:
                                        imp = False
                                        for i in range(1, len(route)-2):
                                            for k in range(i+1, len(route)-1):
                                                new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                                                if route_length(new_route) < route_length(route) - 1e-12:
                                                    route = new_route
                                                    imp = True
                                                    break
                                            if imp:
                                                break
                                    routes[r] = route
                            current_max = max_route_len(routes)
                            improved_this_iter = True

                elif nh == 'intra_2opt':
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        best_delta = 0.0
                        best_ij = None
                        for i in range(1, len(route)-2):
                            for k in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                                new_len = route_length(new_route)
                                if new_len < route_length(route) - 1e-12:
                                    delta = route_length(route) - new_len
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_ij = (i, k, r_idx)
                        if best_ij is not None:
                            i, k, r_idx = best_ij
                            routes[r_idx] = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            new_max = max_route_len(routes)
                            if new_max < current_max:
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                            improved_this_iter = True
                            break

                elif nh == 'intra_or_opt':
                    # Or-opt: move segments of 1 or 2 customers within a route
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 4:
                            continue
                        best_delta = 0.0
                        best_move = None
                        for seg_len in [1, 2]:
                            for start in range(1, len(route)-seg_len):
                                seg = route[start:start+seg_len]
                                for insert_pos in range(1, len(route)):
                                    if insert_pos == start or insert_pos == start+seg_len:
                                        continue
                                    new_route = route[:start] + route[start+seg_len:]
                                    new_route = new_route[:insert_pos] + seg + new_route[insert_pos:]
                                    new_len = route_length(new_route)
                                    if new_len < route_length(route) - 1e-12:
                                        delta = route_length(route) - new_len
                                        if delta > best_delta:
                                            best_delta = delta
                                            best_move = (r_idx, start, seg_len, insert_pos)
                        if best_move is not None:
                            r_idx, start, seg_len, insert_pos = best_move
                            route = routes[r_idx]
                            seg = route[start:start+seg_len]
                            new_route = route[:start] + route[start+seg_len:]
                            new_route = new_route[:insert_pos] + seg + new_route[insert_pos:]
                            routes[r_idx] = new_route
                            new_max = max_route_len(routes)
                            if new_max < current_max:
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                            improved_this_iter = True
                            break

            if improved_this_iter:
                stagnation = 0
                perturbation_size = max(1, n // 10)
            else:
                stagnation += 1
                if stagnation >= perturbation_threshold:
                    # Ruin-recreate perturbation: biased removal from longest route
                    route_lens = [(route_length(r), idx) for idx, r in enumerate(routes)]
                    route_lens.sort(reverse=True)
                    num_to_remove = min(perturbation_size, n - 1)
                    removed = []
                    # Remove more from longest routes
                    for _, r_idx in route_lens:
                        route = routes[r_idx]
                        if len(route) <= 2:
                            continue
                        can_remove = min(num_to_remove - len(removed), len(route)-2)
                        if can_remove <= 0:
                            break
                        remove_set = set(random.sample(route[1:-1], can_remove))
                        for cust in remove_set:
                            removed.append((r_idx, cust))
                        routes[r_idx] = [x for x in route if x not in remove_set]
                        if len(removed) >= num_to_remove:
                            break
                    # Reinsert using regret-3
                    unassigned = [cust for _, cust in removed]
                    random.shuffle(unassigned)
                    while unassigned:
                        best_cust = None
                        best_regret = -1.0
                        best_data = None
                        for cust in unassigned:
                            insert_info = []
                            for r_idx, route in enumerate(routes):
                                for pos in range(1, len(route)):
                                    prev = route[pos-1]
                                    nxt = route[pos]
                                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                    new_len = route_length(route) + cost
                                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                                    new_max = max(new_len, *other_lens)
                                    insert_info.append((new_max, cost, r_idx, pos))
                            if not insert_info:
                                continue
                            insert_info.sort(key=lambda x: (x[0], x[1]))
                            best = insert_info[0]
                            second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                            third = insert_info[2] if len(insert_info) > 2 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                            regret = (second[0] - best[0]) + (third[0] - best[0])  # regret-3
                            if best_cust is None or regret > best_regret or (regret == best_regret and best[1] < best_data[1]):
                                best_cust = cust
                                best_regret = regret
                                best_data = (best[0], best[1], best[2], best[3])
                        if best_cust is None:
                            break
                        _, _, r_idx, pos = best_data
                        routes[r_idx].insert(pos, best_cust)
                        unassigned.remove(best_cust)
                    current_max = max_route_len(routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                    perturbation_size = min(perturbation_size + max(1, n // 20), n // 4)
                    stagnation = 0

            iter_count += 1

        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    if best_routes is None:
        best_routes = routes
    return best_routes
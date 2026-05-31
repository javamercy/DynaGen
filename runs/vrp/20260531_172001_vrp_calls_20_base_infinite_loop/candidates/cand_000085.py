import numpy as np
import random
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def max_route_len(routes):
        return max(route_length(r) for r in routes)

    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 10)

    for attempt in range(max_attempts):
        # Construction: min-max greedy with regret tie-breaking
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
            new_max_val, _, cost_val, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)

        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

        # Improvement phase with adaptive perturbation (exploitation-focused)
        lengths = [route_length(r) for r in routes]
        current_max = max(lengths)
        iter_count = 0
        max_iter = n * truck_count * 2
        stagnation = 0
        perturbation_threshold = 20
        perturb_size = max(1, n // 50)  # start with 2%
        max_perturb_size = max(1, n // 10)  # maximum 10%
        perturb_increment = max(1, n // 50)

        neighborhoods = ['inter_relocate', 'inter_swap', 'intra_2opt']
        nh_success = {nh: 0 for nh in neighborhoods}

        while iter_count < max_iter:
            improved_this_iter = False
            if any(nh_success.values()):
                sorted_nh = sorted(neighborhoods, key=lambda x: -nh_success[x])
            else:
                sorted_nh = neighborhoods
            for nh in sorted_nh:
                if nh == 'intra_2opt':
                    # Sort routes by length descending to prioritize worst routes
                    route_order = sorted(range(truck_count), key=lambda i: -lengths[i])
                    for r_idx in route_order:
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        best_delta = 0
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
                            lengths[r_idx] = route_length(routes[r_idx])
                            new_max = max(lengths)
                            if new_max < current_max:
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                            improved_this_iter = True
                            nh_success[nh] += 1
                            break  # first improvement in this route, move to next neighborhood? But we continue to next route? Actually we break out of route loop to restart neighborhood loop
                        # if improved, break out of route loop to restart neighborhood loop
                        if improved_this_iter:
                            break
                    if improved_this_iter:
                        break
                elif nh == 'inter_relocate':
                    # Focus on longest route
                    lengths = [route_length(r) for r in routes]
                    max_idx = np.argmax(lengths)
                    max_route = routes[max_idx]
                    if len(max_route) > 2:
                        best_delta = 0
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
                                    new_max_candidate = max(new_max_len, new_other_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)])
                                    if new_max_candidate < current_max - 1e-12:
                                        delta = current_max - new_max_candidate
                                        if delta > best_delta:
                                            best_delta = delta
                                            best_move = (cust, max_idx, r_idx, pos)
                        if best_move is not None:
                            cust, from_idx, to_idx, pos = best_move
                            routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                            routes[to_idx].insert(pos, cust)
                            lengths[from_idx] = route_length(routes[from_idx])
                            lengths[to_idx] = route_length(routes[to_idx])
                            current_max = max(lengths)
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                            improved_this_iter = True
                            nh_success[nh] += 1
                            break  # restart neighborhood loop
                elif nh == 'inter_swap':
                    lengths = [route_length(r) for r in routes]
                    max_idx = np.argmax(lengths)
                    max_route = routes[max_idx]
                    if len(max_route) > 2:
                        best_delta = 0
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
                                    new_max_candidate = max(new_max_len, new_other_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, other_idx)])
                                    if new_max_candidate < current_max - 1e-12:
                                        delta = current_max - new_max_candidate
                                        if delta > best_delta:
                                            best_delta = delta
                                            best_move = (cust_i, max_idx, cust_j, other_idx)
                        if best_move is not None:
                            cust_i, from_idx, cust_j, to_idx = best_move
                            routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                            routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                            lengths[from_idx] = route_length(routes[from_idx])
                            lengths[to_idx] = route_length(routes[to_idx])
                            current_max = max(lengths)
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                            improved_this_iter = True
                            nh_success[nh] += 1
                            break  # restart neighborhood loop
                if improved_this_iter:
                    stagnation = 0
                    perturb_size = max(1, n // 50)  # reset perturbation size
                    break

            if not improved_this_iter:
                stagnation += 1
                if stagnation >= perturbation_threshold:
                    # Adaptive perturbation: move perturb_size customers
                    customers = list(range(1, n))
                    random.shuffle(customers)
                    num_perturb = min(perturb_size, len(customers))
                    for cust in customers[:num_perturb]:
                        for r_idx, route in enumerate(routes):
                            if cust in route:
                                routes[r_idx] = [x for x in route if x != cust]
                                break
                        r_idx = random.randrange(truck_count)
                        pos = random.randrange(1, len(routes[r_idx]))
                        routes[r_idx].insert(pos, cust)
                    lengths = [route_length(r) for r in routes]
                    current_max = max(lengths)
                    stagnation = 0
                    perturb_size = min(perturb_size + perturb_increment, max_perturb_size)

            iter_count += 1
            if not improved_this_iter and stagnation > 0:
                if iter_count >= max_iter:
                    break

        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    if best_routes is None:
        best_routes = routes
    return best_routes
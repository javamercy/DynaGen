import numpy as np
import random
from math import exp

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
    max_attempts = max(2, n // 10)

    for _ in range(max_attempts):
        # Construction: min-max greedy with regret-3
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
                # regret-3: difference between best and third best
                if len(insert_info) >= 3:
                    best = insert_info[0][0]
                    third = insert_info[2][0]
                    regret = third - best
                elif len(insert_info) == 2:
                    regret = insert_info[1][0] - insert_info[0][0]
                else:
                    regret = 0
                best_info = insert_info[0]
                candidates.append((best_info[0], regret, best_info[1], best_info[2], best_info[3], cust))
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
        neighborhoods = ['inter_relocate', 'inter_swap', 'intra_2opt']
        nh_success = {nh: 0.0 for nh in neighborhoods}
        nh_attempts = {nh: 0.0 for nh in neighborhoods}
        stagnation = 0
        max_iter = n * truck_count * 2
        iter_count = 0
        perturbation_size = max(1, n // 10)
        perturbation_threshold = 5
        perturbation_increment = max(1, n // 20)
        max_perturbation = n // 4
        shake_threshold = 15

        while iter_count < max_iter:
            # Select neighborhood using softmax on success rates
            if any(nh_attempts.values()):
                success_rates = []
                for nh in neighborhoods:
                    if nh_attempts[nh] > 0:
                        success_rates.append(nh_success[nh] / nh_attempts[nh])
                    else:
                        success_rates.append(0.0)
                temperature = 0.5
                probs = [exp(s/temperature) for s in success_rates]
                total = sum(probs)
                if total > 0:
                    probs = [p/total for p in probs]
                    nh_choice = random.choices(neighborhoods, weights=probs, k=1)[0]
                else:
                    nh_choice = random.choice(neighborhoods)
            else:
                nh_choice = random.choice(neighborhoods)

            improved_this_iter = False

            # Apply chosen neighborhood
            if nh_choice == 'inter_relocate':
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
                                        best_move = (cust, max_idx, r_idx, pos, new_max_candidate)
                    if best_move is not None:
                        cust, from_idx, to_idx, pos, new_max_val = best_move
                        routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                        routes[to_idx].insert(pos, cust)
                        current_max = new_max_val
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
                        improved_this_iter = True

            elif nh_choice == 'inter_swap':
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
                                        best_move = (cust_i, max_idx, cust_j, other_idx, new_max_candidate)
                    if best_move is not None:
                        cust_i, from_idx, cust_j, to_idx, new_max_val = best_move
                        routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                        routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                        current_max = new_max_val
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
                        improved_this_iter = True

            else:  # intra_2opt
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

            # Update success statistics
            nh_attempts[nh_choice] += 1.0
            if improved_this_iter:
                nh_success[nh_choice] += 1.0
                stagnation = 0
                perturbation_size = max(1, n // 10)
            else:
                stagnation += 1
                if stagnation >= perturbation_threshold:
                    # Biased ruin: remove customers from longest routes, prefer high detour cost
                    route_lens = [(route_length(r), idx) for idx, r in enumerate(routes)]
                    route_lens.sort(reverse=True)
                    num_to_remove = min(perturbation_size, n - 1)
                    removed = []
                    for _, r_idx in route_lens:
                        route = routes[r_idx]
                        if len(route) <= 2:
                            continue
                        # Assign bias: detour cost of each customer
                        biased_cust = []
                        for cust in route[1:-1]:
                            prev_idx = route.index(cust) - 1
                            nxt_idx = route.index(cust) + 1
                            detour = distance_matrix[route[prev_idx], cust] + distance_matrix[cust, route[nxt_idx]] - distance_matrix[route[prev_idx], route[nxt_idx]]
                            biased_cust.append((detour, cust))
                        biased_cust.sort(reverse=True)
                        can_remove = min(num_to_remove - len(removed), len(route)-2)
                        if can_remove <= 0:
                            break
                        # Remove customers with highest detour
                        remove_custs = [c for _, c in biased_cust[:can_remove]]
                        for cust in remove_custs:
                            removed.append((r_idx, cust))
                        routes[r_idx] = [x for x in route if x not in remove_custs]
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
                            # regret-3
                            if len(insert_info) >= 3:
                                best_val = insert_info[0][0]
                                third_val = insert_info[2][0]
                                regret = third_val - best_val
                            elif len(insert_info) == 2:
                                regret = insert_info[1][0] - insert_info[0][0]
                            else:
                                regret = 0.0
                            if best_cust is None or regret > best_regret or (regret == best_regret and insert_info[0][1] < best_data[1]):
                                best_cust = cust
                                best_regret = regret
                                best_data = (insert_info[0][0], insert_info[0][1], insert_info[0][2], insert_info[0][3])
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
                    perturbation_size = min(perturbation_size + perturbation_increment, max_perturbation)
                    stagnation = 0
                    nh_success = {nh: 0.0 for nh in neighborhoods}
                    nh_attempts = {nh: 0.0 for nh in neighborhoods}

            # Shake: intra-route random permutation after long stagnation
            if stagnation >= shake_threshold:
                for r_idx in range(truck_count):
                    if len(routes[r_idx]) > 2:
                        interior = routes[r_idx][1:-1]
                        random.shuffle(interior)
                        routes[r_idx] = [0] + interior + [0]
                current_max = max_route_len(routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(routes)
                stagnation = 0
                perturbation_size = max(1, n // 10)
                nh_success = {nh: 0.0 for nh in neighborhoods}
                nh_attempts = {nh: 0.0 for nh in neighborhoods}

            iter_count += 1

        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    if best_routes is None:
        best_routes = routes
    return best_routes
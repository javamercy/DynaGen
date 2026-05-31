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
        return max(route_length(r) for r in routes) if routes else float('inf')

    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 10)

    for _ in range(max_attempts):
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
        nh_success = {nh: 0 for nh in neighborhoods}
        stagnation = 0
        perturb_size = 0.15
        max_perturb_size = 0.30
        perturb_inc = 0.03
        max_iter = n * truck_count * 3
        initial_temp = current_max if current_max > 0 else 1.0
        cooling_rate = 0.99
        iter_count = 0

        while iter_count < max_iter:
            T = initial_temp * (cooling_rate ** iter_count)
            if T < 1e-12:
                T = 1e-12

            # Adaptive softmax temperature
            softmax_temp = max(0.1, 1.0 - iter_count / max_iter)
            if any(nh_success.values()):
                success_vals = [nh_success[nh] for nh in neighborhoods]
                probs = [exp(s / softmax_temp) for s in success_vals]
                total = sum(probs)
                probs = [p / total for p in probs]
                nh_choice = random.choices(neighborhoods, weights=probs, k=1)[0]
            else:
                nh_choice = random.choice(neighborhoods)

            improved_this_iter = False

            if nh_choice == 'inter_relocate':
                # Try best improving move
                lengths = [route_length(r) for r in routes]
                max_idx = int(np.argmax(lengths))
                max_route = routes[max_idx]
                best_delta = 0.0
                best_move = None
                if len(max_route) > 2:
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
                if best_move:
                    cust, from_idx, to_idx, pos, new_max_val = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    current_max = new_max_val
                    improved_this_iter = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                else:
                    # Random move with SA
                    # Pick a random customer from longest route and relocate to random position in another route
                    if len(max_route) > 2:
                        cust = random.choice(max_route[1:-1])
                        other_idx = random.choice([i for i in range(truck_count) if i != max_idx])
                        other_route = routes[other_idx]
                        if len(other_route) >= 2:
                            pos = random.randint(1, len(other_route)-1)
                            # Compute delta
                            new_max_route = [x for x in max_route if x != cust]
                            new_max_len = route_length(new_max_route)
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, other_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                            delta = new_max_candidate - current_max
                            if delta <= 0 or random.random() < exp(-delta / T):
                                routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                                routes[other_idx].insert(pos, cust)
                                current_max = new_max_candidate
                                improved_this_iter = True
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)

            elif nh_choice == 'inter_swap':
                lengths = [route_length(r) for r in routes]
                max_idx = int(np.argmax(lengths))
                max_route = routes[max_idx]
                best_delta = 0.0
                best_move = None
                if len(max_route) > 2:
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
                if best_move:
                    cust_i, from_idx, cust_j, to_idx, new_max_val = best_move
                    routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                    routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                    current_max = new_max_val
                    improved_this_iter = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                else:
                    # Random swap
                    if len(max_route) > 2:
                        other_idx = random.choice([i for i in range(truck_count) if i != max_idx])
                        other_route = routes[other_idx]
                        if len(other_route) > 2:
                            cust_i = random.choice(max_route[1:-1])
                            cust_j = random.choice(other_route[1:-1])
                            new_max_route = [x if x != cust_i else cust_j for x in max_route]
                            new_other_route = [x if x != cust_j else cust_i for x in other_route]
                            new_max_len = route_length(new_max_route)
                            new_other_len = route_length(new_other_route)
                            other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, other_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                            delta = new_max_candidate - current_max
                            if delta <= 0 or random.random() < exp(-delta / T):
                                routes[max_idx] = new_max_route
                                routes[other_idx] = new_other_route
                                current_max = new_max_candidate
                                improved_this_iter = True
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)

            else:  # intra_2opt
                # Try best improving 2-opt on any route
                best_delta = 0.0
                best_ij = None
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for k in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            new_len = route_length(new_route)
                            old_len = route_length(route)
                            if new_len < old_len - 1e-12:
                                delta = old_len - new_len
                                if delta > best_delta:
                                    best_delta = delta
                                    best_ij = (i, k, r_idx)
                if best_ij:
                    i, k, r_idx = best_ij
                    old_route = routes[r_idx]
                    routes[r_idx] = old_route[:i] + old_route[i:k+1][::-1] + old_route[k+1:]
                    current_max = max_route_len(routes)
                    improved_this_iter = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                else:
                    # Random 2-opt on random route
                    if truck_count > 0:
                        r_idx = random.randrange(truck_count)
                        route = routes[r_idx]
                        if len(route) > 3:
                            i = random.randrange(1, len(route)-2)
                            k = random.randrange(i+1, len(route)-1)
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            new_len = route_length(new_route)
                            old_len = route_length(route)
                            delta = new_len - old_len  # route length change, not max
                            # Compute global max after move
                            new_routes = routes[:]
                            new_routes[r_idx] = new_route
                            new_max = max_route_len(new_routes)
                            delta_max = new_max - current_max
                            if delta_max <= 0 or random.random() < exp(-delta_max / T):
                                routes[r_idx] = new_route
                                current_max = new_max
                                improved_this_iter = True
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)

            if improved_this_iter:
                nh_success[nh_choice] += 1
                stagnation = 0
                perturb_size = 0.15
            else:
                stagnation += 1
                if stagnation >= 15:
                    # Ruin-recreate perturbation
                    route_lens = [(route_length(r), idx) for idx, r in enumerate(routes)]
                    route_lens.sort(reverse=True)
                    num_to_remove = max(1, int((n-1) * perturb_size))
                    removed = []
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
                            regret = second[0] - best[0]
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
                    perturb_size = min(perturb_size + perturb_inc, max_perturb_size)
                    stagnation = 0
                    nh_success = {nh: 0 for nh in neighborhoods}

            iter_count += 1

        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    if best_routes is None:
        best_routes = routes
    return best_routes
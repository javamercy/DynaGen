import numpy as np
import random
from math import exp

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
        return max(route_length(r) for r in routes) if routes else float('inf')

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
        candidates.sort(key=lambda x: (x[0], -x[1], x[2], x[5]))
        chosen = candidates[0]
        _, _, _, r_idx, pos, cust = chosen
        routes[r_idx].insert(pos, cust)
        unassigned.remove(cust)

    current_routes = [r[:] for r in routes]
    current_max = max_route_len(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_max = current_max
    report_best_vrp(best_routes)

    # Improvement: Simulated Annealing with fixed cooling
    neighborhoods = ['inter_relocate', 'inter_swap', 'intra_2opt']
    avg_route_len = sum(route_length(r) for r in current_routes) / truck_count
    initial_temp = avg_route_len * 0.1
    if initial_temp < 1e-12:
        initial_temp = 1.0
    cooling_rate = 0.995
    max_iter = n * truck_count * 2
    stagnation = 0
    perturb_size = max(1, int((n-1) * 0.15))
    max_perturb_size = int((n-1) * 0.3)
    perturb_inc = max(1, int((n-1) * 0.05))

    for iteration in range(max_iter):
        T = initial_temp * (cooling_rate ** iteration)
        if T < 1e-12:
            T = 1e-12

        nh_choice = random.choice(neighborhoods)
        improved = False
        new_routes = None
        new_max = None

        if nh_choice == 'inter_relocate':
            lengths = [route_length(r) for r in current_routes]
            max_idx = int(np.argmax(lengths))
            max_route = current_routes[max_idx]
            if len(max_route) > 2:
                best_move = None
                best_delta = 0.0
                for cust in max_route[1:-1]:
                    new_max_route = [x for x in max_route if x != cust]
                    new_max_len = route_length(new_max_route)
                    for r_idx in range(truck_count):
                        if r_idx == max_idx:
                            continue
                        other_route = current_routes[r_idx]
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + [cust] + other_route[pos:]
                            new_other_len = route_length(new_other)
                            other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                            delta = current_max - new_max_candidate
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (cust, max_idx, r_idx, pos, new_max_candidate)
                if best_move:
                    cust, from_idx, to_idx, pos, val = best_move
                    candidate_routes = [r[:] for r in current_routes]
                    candidate_routes[from_idx] = [x for x in candidate_routes[from_idx] if x != cust]
                    candidate_routes[to_idx].insert(pos, cust)
                    candidate_max = max_route_len(candidate_routes)
                    if candidate_max < current_max or random.random() < exp((current_max - candidate_max)/T):
                        current_routes = candidate_routes
                        current_max = candidate_max
                        improved = True
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in current_routes]
                            report_best_vrp(best_routes)

        elif nh_choice == 'inter_swap':
            lengths = [route_length(r) for r in current_routes]
            max_idx = int(np.argmax(lengths))
            max_route = current_routes[max_idx]
            if len(max_route) > 2:
                best_move = None
                best_delta = 0.0
                for cust_i in max_route[1:-1]:
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = current_routes[other_idx]
                        for cust_j in other_route[1:-1]:
                            new_max_route = [x if x != cust_i else cust_j for x in max_route]
                            new_other_route = [x if x != cust_j else cust_i for x in other_route]
                            new_max_len = route_length(new_max_route)
                            new_other_len = route_length(new_other_route)
                            other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, other_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                            delta = current_max - new_max_candidate
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (cust_i, max_idx, cust_j, other_idx, new_max_candidate)
                if best_move:
                    cust_i, from_idx, cust_j, to_idx, val = best_move
                    candidate_routes = [r[:] for r in current_routes]
                    candidate_routes[from_idx] = [x if x != cust_i else cust_j for x in candidate_routes[from_idx]]
                    candidate_routes[to_idx] = [x if x != cust_j else cust_i for x in candidate_routes[to_idx]]
                    candidate_max = max_route_len(candidate_routes)
                    if candidate_max < current_max or random.random() < exp((current_max - candidate_max)/T):
                        current_routes = candidate_routes
                        current_max = candidate_max
                        improved = True
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in current_routes]
                            report_best_vrp(best_routes)

        else:  # intra_2opt
            for r_idx in range(truck_count):
                route = current_routes[r_idx]
                if len(route) <= 3:
                    continue
                best_delta = 0.0
                best_ij = None
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        new_len = route_length(new_route)
                        old_len = route_length(route)
                        delta = old_len - new_len
                        if delta > 1e-12 and delta > best_delta:
                            best_delta = delta
                            best_ij = (i, k, r_idx)
                if best_ij:
                    i, k, r_idx = best_ij
                    candidate_routes = [r[:] for r in current_routes]
                    candidate_routes[r_idx] = candidate_routes[r_idx][:i] + candidate_routes[r_idx][i:k+1][::-1] + candidate_routes[r_idx][k+1:]
                    candidate_max = max_route_len(candidate_routes)
                    if candidate_max < current_max or random.random() < exp((current_max - candidate_max)/T):
                        current_routes = candidate_routes
                        current_max = candidate_max
                        improved = True
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in current_routes]
                            report_best_vrp(best_routes)

        if improved:
            stagnation = 0
            perturb_size = max(1, int((n-1) * 0.15))
        else:
            stagnation += 1
            if stagnation >= 10:
                # Ruin-recreate perturbation
                route_lens = [(route_length(r), idx) for idx, r in enumerate(current_routes)]
                route_lens.sort(reverse=True)
                removed = []
                num_to_remove = min(perturb_size, n-1)
                for _, r_idx in route_lens:
                    route = current_routes[r_idx]
                    if len(route) <= 2:
                        continue
                    can_remove = min(num_to_remove - len(removed), len(route)-2)
                    if can_remove <= 0:
                        break
                    remove_set = set(random.sample(route[1:-1], can_remove))
                    for cust in remove_set:
                        removed.append((r_idx, cust))
                    current_routes[r_idx] = [x for x in route if x not in remove_set]
                    if len(removed) >= num_to_remove:
                        break
                unassigned = [cust for _, cust in removed]
                random.shuffle(unassigned)
                while unassigned:
                    candidates = []
                    for cust in unassigned:
                        insert_info = []
                        for r_idx, route in enumerate(current_routes):
                            for pos in range(1, len(route)):
                                prev = route[pos-1]
                                nxt = route[pos]
                                cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                new_len = route_length(route) + cost
                                other_lens = [route_length(current_routes[i]) for i in range(truck_count) if i != r_idx]
                                new_max = max(new_len, *other_lens)
                                insert_info.append((new_max, cost, r_idx, pos))
                        if not insert_info:
                            continue
                        insert_info.sort(key=lambda x: (x[0], x[1]))
                        best = insert_info[0]
                        second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                        regret = second[0] - best[0]
                        candidates.append((best[0], regret, best[1], best[2], best[3], cust))
                    if not candidates:
                        break
                    candidates.sort(key=lambda x: (x[0], -x[1], x[2], x[5]))
                    chosen = candidates[0]
                    _, _, _, r_idx, pos, cust = chosen
                    current_routes[r_idx].insert(pos, cust)
                    unassigned.remove(cust)
                current_max = max_route_len(current_routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in current_routes]
                    report_best_vrp(best_routes)
                perturb_size = min(perturb_size + perturb_inc, max_perturb_size)
                stagnation = 0

    return best_routes
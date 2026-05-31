import numpy as np
import random

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

    # Single construction: min-max greedy with regret-1 (greedy) tie-breaking
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    while unassigned:
        candidates = []
        for cust in unassigned:
            best_cost = float('inf')
            best_new_max = float('inf')
            best_pos = None
            best_r = None
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if new_max < best_new_max or (new_max == best_new_max and cost < best_cost):
                        best_new_max = new_max
                        best_cost = cost
                        best_pos = pos
                        best_r = r_idx
            if best_r is not None:
                candidates.append((best_new_max, best_cost, best_r, best_pos, cust))
        candidates.sort(key=lambda x: (x[0], x[1]))
        chosen = candidates[0]
        _, _, r_idx, pos, cust = chosen
        routes[r_idx].insert(pos, cust)
        unassigned.remove(cust)

    current_max = max_route_len(routes)
    best_max = current_max
    best_routes = [r[:] for r in routes]
    report_best_vrp(routes)

    # Improvement phase with bounded iterations
    max_iter = n * truck_count * 2
    iteration = 0
    stagnation = 0
    perturbation_size = max(1, n // 10)
    perturbation_threshold = 5
    perturbation_increment = max(1, n // 20)
    max_perturbation = n // 4

    while iteration < max_iter and best_max > 1e-9:
        improved = False
        # VND in fixed order: inter_relocate, inter_swap, intra_2opt
        for nh in ['inter_relocate', 'inter_swap', 'intra_2opt']:
            if improved:
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
                                new_other_route = other_route[:pos] + [cust] + other_route[pos:]
                                new_other_len = route_length(new_other_route)
                                other_lens = [route_length(routes[i]) for i in range(truck_count) if i not in (max_idx, r_idx)]
                                new_max = max(new_max_len, new_other_len, *other_lens)
                                if new_max < current_max - 1e-12:
                                    delta = current_max - new_max
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_move = (cust, max_idx, r_idx, pos, new_max_route, new_other_route)
                    if best_move is not None:
                        cust, from_idx, to_idx, pos, new_max_route, new_other_route = best_move
                        routes[from_idx] = new_max_route
                        routes[to_idx] = new_other_route
                        current_max = current_max - best_delta
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
                        improved = True
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
                                new_max_route = [cust_j if x == cust_i else x for x in max_route]
                                new_other_route = [cust_i if x == cust_j else x for x in other_route]
                                new_max_len = route_length(new_max_route)
                                new_other_len = route_length(new_other_route)
                                other_lens = [route_length(routes[i]) for i in range(truck_count) if i not in (max_idx, other_idx)]
                                new_max = max(new_max_len, new_other_len, *other_lens)
                                if new_max < current_max - 1e-12:
                                    delta = current_max - new_max
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_move = (max_idx, other_idx, new_max_route, new_other_route)
                    if best_move is not None:
                        from_idx, to_idx, new_max_route, new_other_route = best_move
                        routes[from_idx] = new_max_route
                        routes[to_idx] = new_other_route
                        current_max = current_max - best_delta
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
                        improved = True
            else:  # intra_2opt
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    best_delta = 0.0
                    best_move = None
                    for i in range(1, len(route)-2):
                        for k in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            new_len = route_length(new_route)
                            if new_len < route_length(route) - 1e-12:
                                delta = route_length(route) - new_len
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (r_idx, new_route)
                    if best_move is not None:
                        r_idx, new_route = best_move
                        routes[r_idx] = new_route
                        current_max = max_route_len(routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
                        improved = True
                        break
        if improved:
            stagnation = 0
            perturbation_size = max(1, n // 10)
        else:
            stagnation += 1
            if stagnation >= perturbation_threshold:
                # Ruin-recreate with greedy reinsertion (regret-1)
                route_lens = [(route_length(r), idx) for idx, r in enumerate(routes)]
                route_lens.sort(reverse=True)
                num_to_remove = min(perturbation_size, n - 1)
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
                    best_data = None
                    best_cost = float('inf')
                    best_max_val = float('inf')
                    for cust in unassigned:
                        for r_idx, route in enumerate(routes):
                            for pos in range(1, len(route)):
                                prev = route[pos-1]
                                nxt = route[pos]
                                cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                new_len = route_length(route) + cost
                                other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                                new_max = max(new_len, *other_lens)
                                if new_max < best_max_val or (new_max == best_max_val and cost < best_cost):
                                    best_max_val = new_max
                                    best_cost = cost
                                    best_data = (r_idx, pos, cust)
                    if best_data is None:
                        break
                    r_idx, pos, cust = best_data
                    routes[r_idx].insert(pos, cust)
                    unassigned.remove(cust)
                current_max = max_route_len(routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(routes)
                perturbation_size = min(perturbation_size + perturbation_increment, max_perturbation)
                stagnation = 0
        iteration += 1

    if best_routes is None:
        best_routes = routes
    return best_routes
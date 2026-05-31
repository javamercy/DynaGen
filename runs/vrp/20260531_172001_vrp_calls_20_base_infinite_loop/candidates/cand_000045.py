import numpy as np
import random

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

    def construct_solution():
        # min-max greedy with regret tie-breaking
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
        return routes

    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 10)

    for attempt in range(max_attempts):
        # Construction
        routes = construct_solution()
        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

        # Improvement phase
        lengths = [route_length(r) for r in routes]
        improved = True
        iter_count = 0
        max_iter = n * truck_count * 2
        stagnation = 0
        perturbation_threshold = 10

        neighborhoods = [
            'inter_relocate',
            'inter_swap',
            'intra_2opt',
            'intra_relocate'
        ]

        while iter_count < max_iter:
            improved_this_iter = False
            for nh in neighborhoods:
                if nh == 'inter_relocate':
                    improved = False
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
                            current_max = current_max - best_delta
                            improved = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                elif nh == 'inter_swap':
                    improved = False
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
                            current_max = current_max - best_delta
                            improved = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                elif nh == 'intra_2opt':
                    improved = False
                    for r_idx in range(truck_count):
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
                            current_max = max_route_len(routes)
                            improved = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                elif nh == 'intra_relocate':
                    improved = False
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        best_delta = 0
                        best_move = None
                        for pos in range(1, len(route)-1):
                            cust = route[pos]
                            new_route = route[:pos] + route[pos+1:]
                            for new_pos in range(1, len(new_route)):
                                trial = new_route[:new_pos] + [cust] + new_route[new_pos:]
                                new_len = route_length(trial)
                                if new_len < route_length(route) - 1e-12:
                                    delta = route_length(route) - new_len
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_move = (r_idx, pos, new_pos, cust)
                        if best_move is not None:
                            r_idx, old_pos, new_pos, cust = best_move
                            route = routes[r_idx]
                            new_route = route[:old_pos] + route[old_pos+1:]
                            new_route = new_route[:new_pos] + [cust] + new_route[new_pos:]
                            routes[r_idx] = new_route
                            current_max = max_route_len(routes)
                            improved = True
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)

                if improved:
                    improved_this_iter = True
                    stagnation = 0
                    break  # restart neighborhood loop from first

            if not improved_this_iter:
                stagnation += 1
                if stagnation >= perturbation_threshold:
                    # Restart: generate a new solution using construction
                    new_routes = construct_solution()
                    new_max = max_route_len(new_routes)
                    # Accept new solution unconditionally
                    routes = new_routes
                    current_max = new_max
                    stagnation = 0
                    improved_this_iter = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)

            iter_count += 1
            if not improved_this_iter:
                break

        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    if best_routes is None:
        best_routes = routes
    return best_routes
import numpy as np
import random

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
        # Deterministic regret construction
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            best_customer = None
            best_regret = -1.0
            best_data = None
            for cust in sorted(unassigned):
                insert_options = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_options.append((new_max, cost, r_idx, pos))
                if not insert_options:
                    continue
                insert_options.sort(key=lambda x: (x[0], x[1]))
                best = insert_options[0]
                second = insert_options[1] if len(insert_options) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                # Tie-breaking: higher regret first, then lower cost, then smaller customer index
                key = (regret, -best[1], cust)
                if best_customer is None or key > (best_regret, -best_data[1], best_customer):
                    best_customer = cust
                    best_regret = regret
                    best_data = (best[0], best[1], best[2], best[3])
            if best_customer is None:
                break
            _, _, r_idx, pos = best_data
            routes[r_idx].insert(pos, best_customer)
            unassigned.remove(best_customer)

        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

        # Local search
        improved = True
        iter_count = 0
        max_iter = max(1, n * truck_count // 10)  # Reduced iterations to avoid timeout
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            # Inter-route relocate from longest route
            if len(max_route) > 2:
                best_delta = 0.0
                best_move = None
                for i, cust in enumerate(max_route[1:-1]):
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
                            if new_max_candidate < current_max:
                                delta = current_max - new_max_candidate
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (cust, max_idx, r_idx, pos, new_max_candidate)
                if best_move:
                    cust, from_idx, to_idx, pos, new_max_val = best_move
                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                    routes[to_idx].insert(pos, cust)
                    current_max = new_max_val
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
            # Intra-route 2-opt
            if not improved:
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for k in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            if route_length(new_route) < route_length(route):
                                route[:] = new_route
                                improved = True
                                current_max = max_route_len(routes)
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
            # If no improvement, ruin-and-recreate once
            if not improved:
                # Deterministic ruin: remove 20% from longest routes (sorted by length)
                customers_to_remove = []
                sorted_indices = sorted(range(truck_count), key=lambda i: route_length(routes[i]), reverse=True)
                for r_idx in sorted_indices:
                    route = routes[r_idx]
                    if len(route) <= 2:
                        continue
                    num_remove = max(1, int(len(route[1:-1]) * 0.2))
                    # Remove first num_remove customers in order (deterministic)
                    remove_set = set(route[1:1+num_remove])
                    for cust in remove_set:
                        customers_to_remove.append((r_idx, cust))
                    if len(customers_to_remove) >= max(1, (n-1)//5):
                        break
                for r_idx, cust in customers_to_remove:
                    routes[r_idx] = [x for x in routes[r_idx] if x != cust]
                unassigned = [cust for _, cust in customers_to_remove]
                # Deterministic reinsertion (sorted by regret)
                while unassigned:
                    best_customer = None
                    best_regret = -1.0
                    best_data = None
                    # Iterate in sorted order for determinism
                    for cust in sorted(unassigned):
                        insert_options = []
                        for r_idx, route in enumerate(routes):
                            for pos in range(1, len(route)):
                                prev = route[pos-1]
                                nxt = route[pos]
                                cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                                new_len = route_length(route) + cost
                                other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                                new_max = max(new_len, *other_lens)
                                insert_options.append((new_max, cost, r_idx, pos))
                        if not insert_options:
                            continue
                        insert_options.sort(key=lambda x: (x[0], x[1]))
                        best = insert_options[0]
                        second = insert_options[1] if len(insert_options) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                        regret = second[0] - best[0]
                        key = (regret, -best[1], cust)
                        if best_customer is None or key > (best_regret, -best_data[1], best_customer):
                            best_customer = cust
                            best_regret = regret
                            best_data = (best[0], best[1], best[2], best[3])
                    if best_customer is None:
                        break
                    _, _, r_idx, pos = best_data
                    routes[r_idx].insert(pos, best_customer)
                    unassigned.remove(best_customer)
                current_max = max_route_len(routes)
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(routes)

    return best_routes if best_routes else routes
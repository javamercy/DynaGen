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

    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 8)
    # Adaptive parameters
    rand_insert_prob_start = 0.1
    rand_insert_prob_end = 0.0

    for attempt in range(max_attempts):
        # Decay random insertion probability linearly
        rand_insert_prob = rand_insert_prob_start + (rand_insert_prob_end - rand_insert_prob_start) * attempt / max_attempts
        # Regret selection: at start more exploration (choose among top 5), later more greedy (choose among top 2)
        top_k = max(2, 5 - int(attempt * 3 / max_attempts))

        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            if random.random() < rand_insert_prob:
                cust = random.choice(list(unassigned))
                best_cost = float('inf')
                best_r_idx = 0
                best_pos = 1
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if cost < best_cost:
                            best_cost = cost
                            best_r_idx = r_idx
                            best_pos = pos
                routes[best_r_idx].insert(best_pos, cust)
                unassigned.remove(cust)
                continue

            candidates = []
            for cust in unassigned:
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
                insert_options.sort(key=lambda x: (x[0], x[1]))
                best = insert_options[0]
                second = insert_options[1] if len(insert_options) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            # Adaptive selection: choose from top_k
            k = min(top_k, len(candidates))
            chosen = random.choice(candidates[:k])
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)

        current_max = max_route_len(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

        # Local search with adaptive ruin-and-recreate
        improved = True
        iter_count = 0
        max_iter = n * truck_count
        no_improve_count = 0
        no_improve_threshold = max(5, n // 10)  # adaptive threshold
        # Adaptive ruin size
        base_ruin_pct = 0.2
        max_ruin_pct = 0.3

        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            if len(max_route) > 2:
                best_delta = 0
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
                            others = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                            new_max_candidate = max(new_max_len, new_other_len, *others)
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
                    no_improve_count = 0
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)

            if not improved:
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    improved_intra = False
                    for i in range(1, len(route)-2):
                        for k in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            if route_length(new_route) < route_length(route):
                                route[:] = new_route
                                improved_intra = True
                                break
                        if improved_intra:
                            break
                    if improved_intra:
                        improved = True
                        no_improve_count = 0
                        new_max = max_route_len(routes)
                        if new_max < current_max:
                            current_max = new_max
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(routes)
                        break

            if not improved:
                no_improve_count += 1
                if no_improve_count >= no_improve_threshold:
                    # Adaptive ruin size: increase if stuck longer
                    extra = min(0.1, (no_improve_count - no_improve_threshold) * 0.02)
                    ruin_pct = min(max_ruin_pct, base_ruin_pct + extra)
                    customers_to_remove = []
                    for r_idx in sorted(range(truck_count), key=lambda i: route_length(routes[i]), reverse=True):
                        route = routes[r_idx]
                        if len(route) <= 2:
                            continue
                        num_remove = max(1, int(len(route[1:-1]) * ruin_pct))
                        remove_set = set(random.sample(route[1:-1], min(num_remove, len(route[1:-1]))))
                        for cust in remove_set:
                            customers_to_remove.append((r_idx, cust))
                        if len(customers_to_remove) >= n // 5:
                            break
                    for r_idx, cust in customers_to_remove:
                        routes[r_idx] = [x for x in routes[r_idx] if x != cust]
                    unassigned = [cust for _, cust in customers_to_remove]
                    random.shuffle(unassigned)
                    while unassigned:
                        cand_list = []
                        for cust in unassigned:
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
                            insert_options.sort(key=lambda x: (x[0], x[1]))
                            best = insert_options[0]
                            second = insert_options[1] if len(insert_options) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                            regret = second[0] - best[0]
                            cand_list.append((best[0], regret, best[1], best[2], best[3], cust))
                        cand_list.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
                        chosen = cand_list[0]
                        _, _, _, r_idx, pos, cust = chosen
                        routes[r_idx].insert(pos, cust)
                        unassigned.remove(cust)
                    current_max = max_route_len(routes)
                    no_improve_count = 0
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    if best_routes is None:
        best_routes = routes
    return best_routes
import numpy as np
import random
import math

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

    best_routes = None
    best_max = float('inf')
    max_attempts = max(1, n // 10)

    for attempt in range(max_attempts):
        # Construction: min-max greedy with regret tie-breaking (same as parents)
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

        # Tabu search improvement
        tabu = [[0] * truck_count for _ in range(n)]  # tabu[c][r] = iteration when free
        tenure = 10
        no_improve = 0
        max_iter = n * truck_count * 2
        iter_count = 0
        while iter_count < max_iter:
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            candidates = []
            # Inter-relocate moves from max route
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
                        candidates.append(('relocate', new_max_candidate, cust, max_idx, r_idx, pos))
            # Inter-swap moves from max route
            for cust_i in max_route[1:-1]:
                for r_idx in range(truck_count):
                    if r_idx == max_idx:
                        continue
                    other_route = routes[r_idx]
                    for cust_j in other_route[1:-1]:
                        new_max_route = [x if x != cust_i else cust_j for x in max_route]
                        new_other_route = [x if x != cust_j else cust_i for x in other_route]
                        new_max_len = route_length(new_max_route)
                        new_other_len = route_length(new_other_route)
                        other_lens = [lengths[i] for i in range(truck_count) if i not in (max_idx, r_idx)]
                        new_max_candidate = max(new_max_len, new_other_len, *other_lens)
                        candidates.append(('swap', new_max_candidate, cust_i, max_idx, cust_j, r_idx))
            if not candidates:
                break
            # Sort candidates by new_max (ascending)
            candidates.sort(key=lambda x: (x[1], random.random()))
            # Select best move that is non-tabu or satisfies aspiration
            chosen_move = None
            for c in candidates:
                if c[0] == 'relocate':
                    _, new_max, cust, from_idx, to_idx, pos = c
                    if (tabu[cust][from_idx] <= iter_count) or (new_max < best_max - 1e-12):
                        chosen_move = c
                        break
                else:  # swap
                    _, new_max, cust_i, from_idx, cust_j, to_idx = c
                    if (tabu[cust_i][from_idx] <= iter_count and tabu[cust_j][to_idx] <= iter_count) or (new_max < best_max - 1e-12):
                        chosen_move = c
                        break
            if chosen_move is None:
                # All moves are tabu and none improve best; pick the best (first) anyway
                chosen_move = candidates[0]
            # Apply the chosen move
            if chosen_move[0] == 'relocate':
                _, new_max, cust, from_idx, to_idx, pos = chosen_move
                routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                routes[to_idx].insert(pos, cust)
                tabu[cust][from_idx] = iter_count + tenure
            else:  # swap
                _, new_max, cust_i, from_idx, cust_j, to_idx = chosen_move
                routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                tabu[cust_i][from_idx] = iter_count + tenure
                tabu[cust_j][to_idx] = iter_count + tenure
            current_max = new_max
            # Update best
            if current_max < best_max - 1e-12:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(routes)
                no_improve = 0
            else:
                no_improve += 1
            # Intra-route 2-opt improvement (greedy best improvement, no tabu)
            intra_improved = True
            intra_limit = 10  # prevent infinite loop
            intra_iter = 0
            while intra_improved and intra_iter < intra_limit:
                intra_improved = False
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
                    if best_ij:
                        i, k, r_idx = best_ij
                        routes[r_idx] = route[:i] + route[i:k+1][::-1] + route[k+1:]
                        intra_improved = True
                        break  # apply one move per pass
                if intra_improved:
                    new_max = max_route_len(routes)
                    if new_max < current_max:
                        current_max = new_max
                        if current_max < best_max - 1e-12:
                            best_max = current_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(routes)
                            no_improve = 0
                intra_iter += 1
            # Adaptive tabu tenure
            if no_improve > 0 and no_improve % 5 == 0:
                tenure = min(tenure + 1, 20)
            else:
                tenure = max(tenure - 1, 5)
            # Perturbation if stuck
            if no_improve >= 10:
                route_lens = [(route_length(r), idx) for idx, r in enumerate(routes)]
                route_lens.sort(reverse=True)
                num_to_remove = max(1, int((n-1) * 0.10))
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
                if current_max < best_max - 1e-12:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(routes)
                no_improve = 0
                # Reset tabu? Keep tabu to maintain memory.
            iter_count += 1

        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)

    if best_routes is None:
        best_routes = routes
    return best_routes
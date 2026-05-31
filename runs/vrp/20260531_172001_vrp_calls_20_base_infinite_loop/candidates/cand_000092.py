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
        # Min-max regret construction
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
                if len(insert_info) > 1:
                    second = insert_info[1]
                    regret = second[0] - best[0]
                else:
                    regret = 1e9
                candidates.append((best[0], -regret, -best[1], best[2], best[3], cust))
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
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
        max_iter = n * truck_count * 2
        iter_count = 0
        stagnation = 0
        perturbation_size = max(1, n // 10)
        while iter_count < max_iter:
            improved = False

            # Intra-route 2-opt
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
                        delta = route_length(route) - new_len
                        if delta > best_delta + 1e-12:
                            best_delta = delta
                            best_ij = (i, k, r_idx)
                if best_ij is not None:
                    i, k, r_idx = best_ij
                    routes[r_idx] = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    current_max = max_route_len(routes)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(routes)
                    improved = True
                    break  # one improvement per iteration

            if improved:
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= 5:
                    # Ruin-recreate perturbation
                    # Remove random customers from longest route
                    route_lens = [(route_length(r), idx) for idx, r in enumerate(routes)]
                    route_lens.sort(reverse=True)
                    max_route_idx = route_lens[0][1]
                    max_route = routes[max_route_idx]
                    if len(max_route) > 2:
                        num_remove = min(perturbation_size, len(max_route)-2)
                        removed_custs = random.sample(max_route[1:-1], num_remove)
                        routes[max_route_idx] = [x for x in max_route if x not in removed_custs]
                        unassigned = list(removed_custs)
                        random.shuffle(unassigned)
                        while unassigned:
                            best_regret = -1.0
                            best_data = None
                            best_cust = None
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
                                if not insert_info:
                                    continue
                                best = insert_info[0]
                                if len(insert_info) > 1:
                                    second = insert_info[1]
                                    regret = second[0] - best[0]
                                else:
                                    regret = 1e9
                                if best_regret < regret or (best_regret == regret and best[1] < best_data[1]):
                                    best_regret = regret
                                    best_data = (best[0], best[1], best[2], best[3])
                                    best_cust = cust
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
                    stagnation = 0
            iter_count += 1

    if best_routes is None:
        best_routes = routes
    return best_routes
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

    # Greedy construction with regret heuristic
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
            if not insert_info:
                continue
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
    best_routes = [r[:] for r in routes]
    best_max = current_max
    report_best_vrp(best_routes)

    # Simulated annealing parameters (fixed)
    initial_temp = current_max * 0.15
    if initial_temp < 1e-12:
        initial_temp = 1.0
    cooling_rate = 0.985
    max_iter = n * truck_count * 2
    neighborhoods = ['inter_relocate', 'inter_swap', 'intra_2opt']

    for iteration in range(max_iter):
        T = initial_temp * (cooling_rate ** iteration)
        if T < 1e-12:
            T = 1e-12
        nh_choice = random.choice(neighborhoods)
        improved = False

        if nh_choice == 'inter_relocate':
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            if len(max_route) > 2:
                moves = []
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
                            delta = new_max_candidate - current_max
                            moves.append((delta, cust, max_idx, r_idx, pos, new_max_candidate))
                if moves:
                    moves.sort(key=lambda x: x[0])
                    best_move = moves[0]
                    if best_move[0] < 0:
                        _, cust, from_idx, to_idx, pos, new_max_val = best_move
                        routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                        routes[to_idx].insert(pos, cust)
                        current_max = new_max_val
                        improved = True
                    else:
                        for move in moves:
                            if move[0] >= 0:
                                prob = exp(-move[0] / T)
                                if random.random() < prob:
                                    _, cust, from_idx, to_idx, pos, new_max_val = move
                                    routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                                    routes[to_idx].insert(pos, cust)
                                    current_max = new_max_val
                                    improved = True
                                    break
        elif nh_choice == 'inter_swap':
            lengths = [route_length(r) for r in routes]
            max_idx = int(np.argmax(lengths))
            max_route = routes[max_idx]
            if len(max_route) > 2:
                moves = []
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
                            delta = new_max_candidate - current_max
                            moves.append((delta, cust_i, max_idx, cust_j, other_idx, new_max_candidate))
                if moves:
                    moves.sort(key=lambda x: x[0])
                    best_move = moves[0]
                    if best_move[0] < 0:
                        _, cust_i, from_idx, cust_j, to_idx, new_max_val = best_move
                        routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                        routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                        current_max = new_max_val
                        improved = True
                    else:
                        for move in moves:
                            if move[0] >= 0:
                                prob = exp(-move[0] / T)
                                if random.random() < prob:
                                    _, cust_i, from_idx, cust_j, to_idx, new_max_val = move
                                    routes[from_idx] = [x if x != cust_i else cust_j for x in routes[from_idx]]
                                    routes[to_idx] = [x if x != cust_j else cust_i for x in routes[to_idx]]
                                    current_max = new_max_val
                                    improved = True
                                    break
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
                        delta = new_len - route_length(route)
                        if delta < best_delta:
                            best_delta = delta
                            best_ij = (i, k, r_idx)
                if best_ij:
                    i, k, r_idx = best_ij
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    routes[r_idx] = new_route
                    new_max = max_route_len(routes)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                    else:
                        delta = new_max - current_max
                        if random.random() < exp(-delta / T):
                            current_max = new_max
                            improved = True
                        else:
                            # revert
                            routes[r_idx] = route

        if improved:
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)

    return best_routes
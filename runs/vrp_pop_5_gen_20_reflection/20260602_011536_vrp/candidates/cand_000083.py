import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()

    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def objective(routes):
        return max(route_distance(r) for r in routes)

    # Minimax construction
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    while unassigned:
        best_max = float('inf')
        best_total = float('inf')
        best_node = None
        best_route = None
        best_pos = None
        for node in unassigned:
            for r in range(truck_count):
                route = routes[r]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [node] + route[pos:]
                    new_dist = route_distance(new_route)
                    current_max = max(route_distance(routes[rr]) if rr != r else new_dist for rr in range(truck_count))
                    if (current_max < best_max) or (current_max == best_max and new_dist < best_total):
                        best_max = current_max
                        best_total = new_dist
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)

    max_iter = min(30, 2 * n)  # reduce to prevent timeout
    T_start = 5.0
    T_end = 0.1
    routes = [list(r) for r in best_routes]
    routes_obj = best_obj
    removal_frac = 0.3

    for it in range(max_iter):
        current_routes = [list(r) for r in routes]
        # Random removal
        all_customers = [node for route in current_routes for node in route[1:-1]]
        if len(all_customers) == 0:
            continue
        remove_count = max(1, int(removal_frac * (n-1)))
        remove_count = min(remove_count, len(all_customers))
        to_remove = set(random.sample(all_customers, remove_count))
        removed_list = []
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            new_route = [route[0]]
            for node in route[1:-1]:
                if node in to_remove:
                    removed_list.append(node)
                else:
                    new_route.append(node)
            new_route.append(0)
            current_routes[r_idx] = new_route
            if len(current_routes[r_idx]) < 2:
                current_routes[r_idx] = [0, 0]
        random.shuffle(removed_list)

        # Reconstruct with minimax and distance-minimizing tie-breaking
        unassigned = removed_list
        while unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_node = None
            best_route = None
            best_pos = None
            for node in unassigned:
                for r in range(truck_count):
                    route = current_routes[r]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [node] + route[pos:]
                        new_dist = route_distance(new_route)
                        new_max = max(route_distance(current_routes[rr]) if rr != r else new_dist for rr in range(truck_count))
                        if (new_max < best_max) or (new_max == best_max and new_dist < best_total):
                            best_max = new_max
                            best_total = new_dist
                            best_node = node
                            best_route = r
                            best_pos = pos
            if best_node is None:
                break
            current_routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)

        # Best-improvement inter-route relocate (bounded)
        for _ in range(10):
            best_delta = 0
            best_move = None
            current_obj = objective(current_routes)
            # relocate moves
            for i in range(truck_count):
                if len(current_routes[i]) <= 2:
                    continue
                for ci_idx in range(1, len(current_routes[i])-1):
                    ci = current_routes[i][ci_idx]
                    for j in range(truck_count):
                        if i == j:
                            continue
                        for cj_idx in range(1, len(current_routes[j])):  # FIX: exclude appending at end
                            new_route_i = current_routes[i][:ci_idx] + current_routes[i][ci_idx+1:]
                            new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx:]
                            new_max = max(route_distance(new_route_i), route_distance(new_route_j))
                            for k in range(truck_count):
                                if k != i and k != j:
                                    new_max = max(new_max, route_distance(current_routes[k]))
                            delta = current_obj - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('relocate', i, ci_idx, j, cj_idx)
            # swap moves
            for i in range(truck_count):
                if len(current_routes[i]) <= 2:
                    continue
                for ci_idx in range(1, len(current_routes[i])-1):
                    ci = current_routes[i][ci_idx]
                    for j in range(i+1, truck_count):
                        if len(current_routes[j]) <= 2:
                            continue
                        for cj_idx in range(1, len(current_routes[j])-1):
                            cj = current_routes[j][cj_idx]
                            new_route_i = current_routes[i][:ci_idx] + [cj] + current_routes[i][ci_idx+1:]
                            new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx+1:]
                            new_max = max(route_distance(new_route_i), route_distance(new_route_j))
                            for k in range(truck_count):
                                if k != i and k != j:
                                    new_max = max(new_max, route_distance(current_routes[k]))
                            delta = current_obj - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('swap', i, ci_idx, j, cj_idx)
            if best_delta > 0:
                if best_move[0] == 'relocate':
                    _, i, ci_idx, j, cj_idx = best_move
                    ci = current_routes[i][ci_idx]
                    new_route_i = current_routes[i][:ci_idx] + current_routes[i][ci_idx+1:]
                    new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx:]
                    current_routes[i] = new_route_i
                    current_routes[j] = new_route_j
                else:  # swap
                    _, i, ci_idx, j, cj_idx = best_move
                    ci = current_routes[i][ci_idx]
                    cj = current_routes[j][cj_idx]
                    current_routes[i] = current_routes[i][:ci_idx] + [cj] + current_routes[i][ci_idx+1:]
                    current_routes[j] = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx+1:]
            else:
                break

        # Best-improvement intra-route 2-opt (bounded)
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            for _ in range(10):
                best_gain = 0
                best_ij = None
                old_d = route_distance(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_d = route_distance(new_route)
                        gain = old_d - new_d
                        if gain > best_gain:
                            best_gain = gain
                            best_ij = (i, j)
                if best_gain > 0:
                    i, j = best_ij
                    route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    old_d = route_distance(route)
                else:
                    break
            current_routes[r_idx] = route

        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
        # Simulated annealing acceptance
        T = T_start * (T_end / T_start) ** (it / max_iter)
        delta = new_obj - routes_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            routes = [list(r) for r in current_routes]
            routes_obj = new_obj

    return best_routes
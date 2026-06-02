import numpy as np
import random
import math

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
                    new_route_dist = route_distance(new_route)
                    new_max = max(route_distance(routes[rr]) if rr != r else new_route_dist for rr in range(truck_count))
                    if (new_max < best_max) or (new_max == best_max and new_route_dist < best_total):
                        best_max = new_max
                        best_total = new_route_dist
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)

    max_iter = min(50, 2 * n)
    T_start = 5.0
    T_end = 0.1
    routes = [list(r) for r in best_routes]
    routes_obj = best_obj

    for it in range(max_iter):
        current_routes = [list(r) for r in routes]
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        # Adaptive removal fractions: decrease over iterations
        progress = it / max_iter
        frac_long_low = 0.25 - 0.1 * progress  # from 0.25 to 0.15
        frac_long_high = 0.35 - 0.1 * progress  # from 0.35 to 0.25
        frac_short_low = 0.15 - 0.1 * progress  # from 0.15 to 0.05
        frac_short_high = 0.25 - 0.1 * progress  # from 0.25 to 0.15
        remove_frac_long = random.uniform(frac_long_low, frac_long_high)
        remove_frac_short = random.uniform(frac_short_low, frac_short_high)
        removed_list = []
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 2:
                continue
            if route_dists[r_idx] >= max_dist * 0.99:
                frac = remove_frac_long
            else:
                frac = remove_frac_short
            remove_count = max(1, int(frac * (len(route)-2)))
            if remove_count >= len(route)-2:
                remove_count = len(route)-2
            customers = route[1:-1]
            if len(customers) == 0:
                continue
            selected = random.sample(customers, min(remove_count, len(customers)))
            removed_list.extend(selected)
            new_route = [route[0]]
            for node in route[1:-1]:
                if node not in selected:
                    new_route.append(node)
            new_route.append(0)
            current_routes[r_idx] = new_route
            if len(current_routes[r_idx]) < 2:
                current_routes[r_idx] = [0, 0]
        random.shuffle(removed_list)

        # Reconstruct with minimax and tie-breaking
        unassigned = removed_list
        while unassigned:
            best_candidates = []
            best_max = float('inf')
            best_route_total = float('inf')
            for node in unassigned:
                for r in range(truck_count):
                    route = current_routes[r]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [node] + route[pos:]
                        new_route_dist = route_distance(new_route)
                        new_max = max(route_distance(current_routes[rr]) if rr != r else new_route_dist for rr in range(truck_count))
                        current_route_dist = route_distance(route)
                        if (new_max < best_max) or (new_max == best_max and new_route_dist < best_route_total):
                            best_max = new_max
                            best_route_total = new_route_dist
                            best_candidates = [(node, r, pos, current_route_dist)]
                        elif new_max == best_max and new_route_dist == best_route_total:
                            best_candidates.append((node, r, pos, current_route_dist))
            if not best_candidates:
                break
            chosen = min(best_candidates, key=lambda x: (x[3], x[1], x[2]))
            node, r, pos, _ = chosen
            current_routes[r].insert(pos, node)
            unassigned.remove(node)

        # Intensified local search on longest route(s)
        for _ in range(3):
            route_dists = [route_distance(r) for r in current_routes]
            max_dist = max(route_dists)
            longest_indices = [i for i, d in enumerate(route_dists) if d >= max_dist * 0.99]
            improved = True
            attempts = 0
            max_attempts = 50
            while improved and attempts < max_attempts:
                improved = False
                attempts += 1
                current_max = objective(current_routes)
                best_delta = 0
                best_move = None
                # Relocate from longest to any
                for i in longest_indices:
                    if len(current_routes[i]) <= 2:
                        continue
                    for ci_idx in range(1, len(current_routes[i])-1):
                        ci = current_routes[i][ci_idx]
                        for j in range(truck_count):
                            if i == j:
                                continue
                            for cj_idx in range(1, len(current_routes[j])):
                                new_route_i = current_routes[i][:ci_idx] + current_routes[i][ci_idx+1:]
                                new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx:]
                                new_max = max(route_distance(new_route_i), route_distance(new_route_j))
                                for k in range(truck_count):
                                    if k != i and k != j:
                                        new_max = max(new_max, route_distance(current_routes[k]))
                                if new_max < current_max:
                                    delta = current_max - new_max
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_move = ('relocate', i, ci_idx, j, cj_idx)
                # Swap involving longest
                for i in longest_indices:
                    if len(current_routes[i]) <= 2:
                        continue
                    for ci_idx in range(1, len(current_routes[i])-1):
                        ci = current_routes[i][ci_idx]
                        for j in range(truck_count):
                            if i == j:
                                continue
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
                                if new_max < current_max:
                                    delta = current_max - new_max
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_move = ('swap', i, ci_idx, j, cj_idx)
                if best_move is not None:
                    if best_move[0] == 'relocate':
                        _, i, ci_idx, j, cj_idx = best_move
                        ci = current_routes[i][ci_idx]
                        del current_routes[i][ci_idx]
                        if len(current_routes[i]) == 1:
                            current_routes[i] = [0, 0]
                        current_routes[j].insert(cj_idx, ci)
                    else:
                        _, i, ci_idx, j, cj_idx = best_move
                        ci = current_routes[i][ci_idx]
                        cj = current_routes[j][cj_idx]
                        current_routes[i][ci_idx] = cj
                        current_routes[j][cj_idx] = ci
                    improved = True
                # Intra-route 2-opt on longest routes
                for r_idx in longest_indices:
                    route = current_routes[r_idx]
                    if len(route) <= 3:
                        continue
                    improved_opt = True
                    for _ in range(10):
                        improved_opt = False
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                if route_distance(new_route) < route_distance(route):
                                    route = new_route
                                    improved_opt = True
                                    break
                            if improved_opt:
                                break
                        if not improved_opt:
                            break
                    current_routes[r_idx] = route

        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
        T = T_start * (T_end / T_start) ** (it / max_iter)
        delta = new_obj - routes_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            routes = [list(r) for r in current_routes]
            routes_obj = new_obj

    return best_routes
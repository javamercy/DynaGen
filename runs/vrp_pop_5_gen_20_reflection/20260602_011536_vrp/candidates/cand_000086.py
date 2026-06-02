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

    # Adaptive removal fractions: start low and increase
    frac_long_start = 0.20
    frac_long_end = 0.35
    frac_short_start = 0.10
    frac_short_end = 0.20

    for it in range(max_iter):
        current_routes = [list(r) for r in routes]
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        # Adaptive adjustment: interpolate removal fractions based on iteration progress
        progress = it / max_iter
        frac_long = frac_long_start + (frac_long_end - frac_long_start) * progress
        frac_short = frac_short_start + (frac_short_end - frac_short_start) * progress
        # Sample fractions uniformly within given ranges
        remove_frac_long = random.uniform(frac_long - 0.05, frac_long + 0.05) if frac_long > 0 else 0.2
        remove_frac_short = random.uniform(frac_short - 0.05, frac_short + 0.05) if frac_short > 0 else 0.1
        # Clamp to ensure valid ranges
        remove_frac_long = max(0.1, min(0.5, remove_frac_long))
        remove_frac_short = max(0.05, min(0.3, remove_frac_short))

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

        # Reconstruct with minimax and tie-breaking, ensure all customers inserted
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
                        if (new_max < best_max) or (new_max == best_max and new_route_dist < best_route_total) or (new_max == best_max and new_route_dist == best_route_total and current_route_dist < (best_candidates[0][3] if best_candidates else float('inf'))):
                            best_max = new_max
                            best_route_total = new_route_dist
                            best_candidates = [(node, r, pos, current_route_dist)]
                        elif new_max == best_max and new_route_dist == best_route_total and current_route_dist == best_candidates[0][3]:
                            best_candidates.append((node, r, pos, current_route_dist))
            # Always pick a candidate; if none, fallback: insert in first route at end
            if not best_candidates:
                # Fallback: insert at the end of the shortest route
                shortest_route = min(range(truck_count), key=lambda r: route_distance(current_routes[r]))
                node = unassigned[0]
                pos = len(current_routes[shortest_route]) - 1
                current_routes[shortest_route].insert(pos, node)
                unassigned.remove(node)
            else:
                chosen = min(best_candidates, key=lambda x: (x[3], x[1], x[2]))
                node, r, pos, _ = chosen
                current_routes[r].insert(pos, node)
                unassigned.remove(node)

        # Intensified local search on the longest route
        for _ in range(3):
            route_dists = [route_distance(r) for r in current_routes]
            max_dist = max(route_dists)
            longest_indices = [i for i, d in enumerate(route_dists) if d >= max_dist * 0.99]
            # Inter-route moves involving longest routes
            improved = True
            attempts = 0
            max_attempts = 50
            while improved and attempts < max_attempts:
                improved = False
                attempts += 1
                current_max = objective(current_routes)
                best_delta = 0
                best_move = None
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
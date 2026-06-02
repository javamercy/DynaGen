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

    # Minimax construction with correct insertion positions (1..len-1)
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
                # positions between first and last depot (exclude 0 and len-1)
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

    max_iter = 20  # bounded iterations
    T_start = 3.0
    T_end = 0.1
    routes = [list(r) for r in best_routes]
    routes_obj = best_obj
    removal_frac = 0.3

    for it in range(max_iter):
        current_routes = [list(r) for r in routes]
        # Collect customers with their route index
        all_customers = [(node, r_idx) for r_idx, route in enumerate(current_routes) for node in route[1:-1]]
        if len(all_customers) == 0:
            continue
        # Identify longest route(s)
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        longest_indices = [i for i, d in enumerate(route_dists) if d == max_dist]
        # Collect candidates from longest routes
        candidates = [(node, r_idx) for (node, r_idx) in all_customers if r_idx in longest_indices]
        # Biased removal: 60% from longest, rest random
        remove_count = max(1, int(removal_frac * (n-1)))
        remove_count = min(remove_count, len(all_customers))
        to_remove = set()
        num_from_long = min(int(0.6 * remove_count), len(candidates))
        if num_from_long > 0:
            chosen = random.sample(candidates, num_from_long)
            to_remove.update(node for node, _ in chosen)
        remaining = remove_count - len(to_remove)
        if remaining > 0:
            remaining_candidates = [node for node, _ in all_customers if node not in to_remove]
            if remaining_candidates:
                extra = set(random.sample(remaining_candidates, min(remaining, len(remaining_candidates))))
                to_remove.update(extra)
        # Remove selected customers
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
            if len(new_route) < 2:
                new_route = [0, 0]
            current_routes[r_idx] = new_route
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

        # Focused local search: limited passes
        for _ in range(2):  # two passes of each move
            # Inter-route relocate (best improvement)
            improved = False
            best_delta = 0
            best_move = None
            current_obj = objective(current_routes)
            for i in range(truck_count):
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
                            delta = current_obj - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('relocate', i, ci_idx, j, cj_idx)
            if best_delta > 0:
                improved = True
                _, i, ci_idx, j, cj_idx = best_move
                ci = current_routes[i][ci_idx]
                new_route_i = current_routes[i][:ci_idx] + current_routes[i][ci_idx+1:]
                new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx:]
                current_routes[i] = new_route_i
                current_routes[j] = new_route_j
            # Inter-route swap (best improvement)
            best_delta = 0
            best_move = None
            current_obj = objective(current_routes)
            for i in range(truck_count):
                if len(current_routes[i]) <= 2:
                    continue
                for ci_idx in range(1, len(current_routes[i])-1):
                    for j in range(i+1, truck_count):
                        if len(current_routes[j]) <= 2:
                            continue
                        for cj_idx in range(1, len(current_routes[j])-1):
                            ci = current_routes[i][ci_idx]
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
                improved = True
                _, i, ci_idx, j, cj_idx = best_move
                ci = current_routes[i][ci_idx]
                cj = current_routes[j][cj_idx]
                current_routes[i] = current_routes[i][:ci_idx] + [cj] + current_routes[i][ci_idx+1:]
                current_routes[j] = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx+1:]
            # Intra-route 2-opt on longest route only
            route_dists = [route_distance(r) for r in current_routes]
            max_dist = max(route_dists)
            longest_idx = route_dists.index(max_dist)
            route = current_routes[longest_idx]
            if len(route) > 3:
                improved_opt = True
                while improved_opt:
                    improved_opt = False
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
                        improved_opt = True
                        i, j = best_ij
                        route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                current_routes[longest_idx] = route
            if improved:
                new_obj = objective(current_routes)
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_routes = [list(r) for r in current_routes]
                    report_best_vrp(best_routes)

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
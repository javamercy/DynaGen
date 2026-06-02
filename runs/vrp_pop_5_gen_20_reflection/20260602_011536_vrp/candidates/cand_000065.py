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
        # Ruin: remove fraction biased to longest route
        remove_frac = random.uniform(0.2, 0.4)
        remove_count = max(1, int(remove_frac * (n-1)))
        all_customers = list(range(1, n))
        # Identify longest route(s)
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        longest_indices = [i for i, d in enumerate(route_dists) if d == max_dist]
        # Determine removals from longest and others
        if longest_indices:
            # 70% from longest route (if multiple, pick one randomly)
            longest_idx = random.choice(longest_indices)
            longest_customers = [c for c in current_routes[longest_idx] if c != 0]
            other_customers = [c for c in all_customers if c not in longest_customers]
            num_from_longest = min(int(0.7 * remove_count), len(longest_customers))
            num_from_others = remove_count - num_from_longest
            removed_from_longest = set(random.sample(longest_customers, num_from_longest))
            removed_from_others = set(random.sample(other_customers, min(num_from_others, len(other_customers))))
            to_remove = removed_from_longest | removed_from_others
        else:
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
                        if new_max < best_max or (new_max == best_max and new_route_dist < best_route_total) or (new_max == best_max and new_route_dist == best_route_total and current_route_dist < best_candidates[0][3] if best_candidates else True):
                            best_max = new_max
                            best_route_total = new_route_dist
                            best_candidates = [(node, r, pos, current_route_dist)]
                        elif new_max == best_max and new_route_dist == best_route_total and current_route_dist == best_candidates[0][3]:
                            best_candidates.append((node, r, pos, current_route_dist))
            if not best_candidates:
                break
            chosen = min(best_candidates, key=lambda x: (x[3], x[1], x[2]))
            node, r, pos, _ = chosen
            current_routes[r].insert(pos, node)
            unassigned.remove(node)

        # Inter-route improvement: relocate and swap (first improvement reducing max)
        improved = True
        attempts = 0
        max_attempts = 20 * truck_count
        while improved and attempts < max_attempts:
            improved = False
            attempts += 1
            best_delta = 0
            best_move = None
            current_max = objective(current_routes)
            # Relocate
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
                            if new_max < current_max:
                                delta = current_max - new_max
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = ('relocate', i, ci_idx, j, cj_idx)
            # Swap
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

        # Intra-route 2-opt with extra iterations on longest route
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        longest_indices = [i for i, d in enumerate(route_dists) if d == max_dist]
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            # Determine iteration count: more for longest route
            if r_idx in longest_indices:
                max_opt_iters = 20
            else:
                max_opt_iters = 10
            improved_opt = True
            for _ in range(max_opt_iters):
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
        # Simulated annealing acceptance
        T = T_start * (T_end / T_start) ** (it / max_iter)
        delta = new_obj - routes_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            routes = [list(r) for r in current_routes]
            routes_obj = new_obj

    return best_routes
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

    # Build initial solution via nearest-neighbor insertion minimizing max
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    while unassigned:
        best_node = None
        best_route = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for node in unassigned:
            for r in range(truck_count):
                route = routes[r]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [node] + route[pos:]
                    new_dist = route_distance(new_route)
                    new_max = max(route_distance(routes[rr]) if rr != r else new_dist for rr in range(truck_count))
                    # Tie-breaking: min new_max, then min new_dist
                    if (new_max < best_max) or (new_max == best_max and new_dist < best_total):
                        best_max = new_max
                        best_total = new_dist
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
        # Ruin: random removal of a fraction of customers
        remove_frac = random.uniform(0.2, 0.4)
        remove_count = max(1, int(remove_frac * (n - 1)))
        all_customers = list(range(1, n))
        to_remove = set(random.sample(all_customers, min(remove_count, len(all_customers))))
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

        # Reconstruct: insert all removed nodes
        unassigned = removed_list
        while unassigned:
            best_node = None
            best_route = None
            best_pos = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            best_current_dist = float('inf')
            for node in unassigned:
                for r in range(truck_count):
                    route = current_routes[r]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [node] + route[pos:]
                        new_dist = route_distance(new_route)
                        new_max = max(route_distance(current_routes[rr]) if rr != r else new_dist for rr in range(truck_count))
                        current_dist = route_distance(route)
                        # Tie-breaking: min new_max, then min new_dist, then min current_dist, then min route index
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_dist < best_new_total) or
                            (new_max == best_new_max and new_dist == best_new_total and current_dist < best_current_dist) or
                            (new_max == best_new_max and new_dist == best_new_total and current_dist == best_current_dist and r < best_route)):
                            best_new_max = new_max
                            best_new_total = new_dist
                            best_current_dist = current_dist
                            best_node = node
                            best_route = r
                            best_pos = pos
            # Fallback: if no candidate found (shouldn't happen), insert first node into first route
            if best_node is None:
                best_node = unassigned[0]
                best_route = 0
                best_pos = 1
            current_routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)

        # Inter-route improvement: relocate and swap (first improvement on max)
        improved = True
        attempts = 0
        max_attempts = 20 * truck_count
        while improved and attempts < max_attempts:
            improved = False
            attempts += 1
            current_max = objective(current_routes)
            # Relocate
            for i in range(truck_count):
                if len(current_routes[i]) <= 2:
                    continue
                for ci_idx in range(1, len(current_routes[i]) - 1):
                    ci = current_routes[i][ci_idx]
                    for j in range(truck_count):
                        if i == j:
                            continue
                        for cj_idx in range(1, len(current_routes[j])):
                            new_route_i = current_routes[i][:ci_idx] + current_routes[i][ci_idx+1:]
                            new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx:]
                            if len(new_route_i) < 2:
                                new_route_i = [0, 0]
                            new_max = max(route_distance(new_route_i), route_distance(new_route_j))
                            for k in range(truck_count):
                                if k != i and k != j:
                                    new_max = max(new_max, route_distance(current_routes[k]))
                            if new_max < current_max:
                                current_routes[i] = new_route_i
                                current_routes[j] = new_route_j
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                # Swap
                for i in range(truck_count):
                    if len(current_routes[i]) <= 2:
                        continue
                    for ci_idx in range(1, len(current_routes[i]) - 1):
                        ci = current_routes[i][ci_idx]
                        for j in range(i+1, truck_count):
                            if len(current_routes[j]) <= 2:
                                continue
                            for cj_idx in range(1, len(current_routes[j]) - 1):
                                cj = current_routes[j][cj_idx]
                                new_route_i = current_routes[i][:ci_idx] + [cj] + current_routes[i][ci_idx+1:]
                                new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx+1:]
                                new_max = max(route_distance(new_route_i), route_distance(new_route_j))
                                for k in range(truck_count):
                                    if k != i and k != j:
                                        new_max = max(new_max, route_distance(current_routes[k]))
                                if new_max < current_max:
                                    current_routes[i] = new_route_i
                                    current_routes[j] = new_route_j
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        # Intra-route 2-opt on all routes
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            improved_opt = True
            while improved_opt:
                improved_opt = False
                for i in range(1, len(route) - 2):
                    for j in range(i+1, len(route) - 1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_distance(new_route) < route_distance(route):
                            route = new_route
                            improved_opt = True
                            break
                    if improved_opt:
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
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
    
    # Construction: minimax cheapest insertion
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    while unassigned:
        best_max = float('inf')
        best_total = float('inf')
        best_node = None
        best_route = -1
        best_pos = -1
        for node in unassigned:
            for r in range(truck_count):
                route = routes[r]
                # Insert between consecutive nodes (including depot at ends)
                for pos in range(1, len(route)):
                    new_dist = route_distance(route[:pos] + [node] + route[pos:])
                    # Compute max after insertion
                    current_max = max(
                        route_distance(routes[rr]) if rr != r else new_dist
                        for rr in range(truck_count)
                    )
                    total = sum(route_distance(routes[rr]) if rr != r else new_dist for rr in range(truck_count))
                    if (current_max < best_max) or (current_max == best_max and total < best_total):
                        best_max = current_max
                        best_total = total
                        best_node = node
                        best_route = r
                        best_pos = pos
        if best_node is None:
            break
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)
    
    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)
    
    # Parameters
    max_iter = min(30, 2 * n)
    removal_frac = 0.2
    T_start = 3.0
    T_end = 0.1
    routes = [list(r) for r in best_routes]
    routes_obj = best_obj
    
    for it in range(max_iter):
        current_routes = [list(r) for r in routes]
        # Collect all customers
        all_customers = []
        for r_idx, route in enumerate(current_routes):
            for node in route[1:-1]:
                all_customers.append((node, r_idx))
        if len(all_customers) == 0:
            continue
        # Biased removal: 40% from longest routes, rest random
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        longest_indices = [i for i, d in enumerate(route_dists) if d == max_dist]
        candidates = [(node, r_idx) for (node, r_idx) in all_customers if r_idx in longest_indices]
        remove_count = max(1, int(removal_frac * (n - 1)))
        remove_count = min(remove_count, len(all_customers))
        to_remove = set()
        # From longest
        num_from_long = min(int(0.4 * remove_count), len(candidates))
        if num_from_long > 0:
            chosen = random.sample(candidates, num_from_long)
            to_remove.update(node for node, _ in chosen)
        # Remaining random from all
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
        # Reconstruct with minimax insertion
        unassigned = removed_list
        while unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_node = None
            best_route = -1
            best_pos = -1
            for node in unassigned:
                for r in range(truck_count):
                    route = current_routes[r]
                    for pos in range(1, len(route)):
                        new_dist = route_distance(route[:pos] + [node] + route[pos:])
                        current_max = max(
                            route_distance(current_routes[rr]) if rr != r else new_dist
                            for rr in range(truck_count)
                        )
                        total = sum(route_distance(current_routes[rr]) if rr != r else new_dist for rr in range(truck_count))
                        if (current_max < best_max) or (current_max == best_max and total < best_total):
                            best_max = current_max
                            best_total = total
                            best_node = node
                            best_route = r
                            best_pos = pos
            if best_node is None:
                break
            current_routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)
        # Local search: one pass each of relocate, swap, then 2-opt
        # Inter-route relocate (first improvement)
        improved = True
        while improved:
            improved = False
            for i in range(truck_count):
                route_i = current_routes[i]
                if len(route_i) <= 2:
                    continue
                for ci_idx in range(1, len(route_i)-1):
                    ci = route_i[ci_idx]
                    for j in range(truck_count):
                        if i == j:
                            continue
                        route_j = current_routes[j]
                        for cj_idx in range(1, len(route_j)+1):
                            new_route_i = route_i[:ci_idx] + route_i[ci_idx+1:]
                            new_route_j = route_j[:cj_idx] + [ci] + route_j[cj_idx:]
                            # Ensure routes valid
                            if len(new_route_i) < 2:
                                new_route_i = [0,0]
                            old_obj = objective(current_routes)
                            # Compute new max quickly
                            new_max = max(route_distance(new_route_i), route_distance(new_route_j))
                            for k in range(truck_count):
                                if k != i and k != j:
                                    new_max = max(new_max, route_distance(current_routes[k]))
                            if new_max < old_obj:
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
            # Inter-route swap (first improvement)
            if not improved:
                # Only try swap if no relocate found
                for i in range(truck_count):
                    route_i = current_routes[i]
                    if len(route_i) <= 2:
                        continue
                    for ci_idx in range(1, len(route_i)-1):
                        ci = route_i[ci_idx]
                        for j in range(i+1, truck_count):
                            route_j = current_routes[j]
                            if len(route_j) <= 2:
                                continue
                            for cj_idx in range(1, len(route_j)-1):
                                cj = route_j[cj_idx]
                                new_route_i = route_i[:ci_idx] + [cj] + route_i[ci_idx+1:]
                                new_route_j = route_j[:cj_idx] + [ci] + route_j[cj_idx+1:]
                                old_obj = objective(current_routes)
                                new_max = max(route_distance(new_route_i), route_distance(new_route_j))
                                for k in range(truck_count):
                                    if k != i and k != j:
                                        new_max = max(new_max, route_distance(current_routes[k]))
                                if new_max < old_obj:
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
        # Intra-route 2-opt (one pass per route)
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
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
            current_routes[r_idx] = route
        # Evaluate
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
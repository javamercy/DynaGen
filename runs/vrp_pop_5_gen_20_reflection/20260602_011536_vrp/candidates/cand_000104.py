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

    # Initialize routes with depot only
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    random.shuffle(unassigned)

    # Greedy insertion: at each step, insert customer to minimize max route distance
    for node in unassigned:
        best_max = float('inf')
        best_total = float('inf')
        best_route = None
        best_pos = None
        for r in range(truck_count):
            route = routes[r]
            for pos in range(1, len(route)):
                new_route = route[:pos] + [node] + route[pos:]
                new_dist = route_distance(new_route)
                # new max distance
                new_max = max(route_distance(routes[rr]) if rr != r else new_dist for rr in range(truck_count))
                if new_max < best_max or (new_max == best_max and new_dist < best_total):
                    best_max = new_max
                    best_total = new_dist
                    best_route = r
                    best_pos = pos
        routes[best_route].insert(best_pos, node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)

    # Iterated local search with bounded iterations
    max_iter = min(10, 2 * n)
    for it in range(max_iter):
        current_routes = [list(r) for r in routes]
        # Remove a small number of customers from the longest route(s)
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        longest_indices = [i for i, d in enumerate(route_dists) if d == max_dist]
        # Collect customers from longest routes
        candidates = []
        for idx in longest_indices:
            route = current_routes[idx]
            for node in route[1:-1]:
                candidates.append((node, idx))
        if not candidates:
            continue
        # Remove 1 to 3 customers (or all if fewer)
        remove_count = min(random.randint(1, 3), len(candidates))
        removed = random.sample(candidates, remove_count)
        removed_nodes = [node for node, _ in removed]
        # Remove from routes
        for node, r_idx in removed:
            route = current_routes[r_idx]
            route.remove(node)
            # Ensure route still has depot at both ends
            if len(route) == 1:
                route = [0, 0]
            else:
                # Redundant but safe
                if route[0] != 0:
                    route.insert(0, 0)
                if route[-1] != 0:
                    route.append(0)
            current_routes[r_idx] = route
        # Shuffle removed nodes order
        random.shuffle(removed_nodes)
        # Reinsert using minimax insertion
        while removed_nodes:
            node = removed_nodes.pop()
            best_max = float('inf')
            best_total = float('inf')
            best_route = None
            best_pos = None
            for r in range(truck_count):
                route = current_routes[r]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [node] + route[pos:]
                    new_dist = route_distance(new_route)
                    new_max = max(route_distance(current_routes[rr]) if rr != r else new_dist for rr in range(truck_count))
                    if new_max < best_max or (new_max == best_max and new_dist < best_total):
                        best_max = new_max
                        best_total = new_dist
                        best_route = r
                        best_pos = pos
            current_routes[best_route].insert(best_pos, node)

        # Local search: one pass of best-improvement relocate and swap (bounded)
        # Relocate
        for _ in range(2 * n):
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
                        for cj_idx in range(1, len(current_routes[j])+1):
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
                _, i, ci_idx, j, cj_idx = best_move
                ci = current_routes[i][ci_idx]
                current_routes[i] = current_routes[i][:ci_idx] + current_routes[i][ci_idx+1:]
                current_routes[j] = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx:]
                improved = True
            if not improved:
                break

        # Swap
        for _ in range(2 * n):
            improved = False
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
                _, i, ci_idx, j, cj_idx = best_move
                ci = current_routes[i][ci_idx]
                cj = current_routes[j][cj_idx]
                current_routes[i] = current_routes[i][:ci_idx] + [cj] + current_routes[i][ci_idx+1:]
                current_routes[j] = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx+1:]
                improved = True
            if not improved:
                break

        # Intra-route 2-opt (single pass per route)
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            improved = True
            while improved:
                improved = False
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
                    improved = True
            current_routes[r_idx] = route

        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
            # Accept new routes unconditionally (since we found better)
            routes = [list(r) for r in current_routes]
        # Simple acceptance: if same or worse, keep current routes (no worse)
        # To avoid stuck, we still update routes to current? Actually we keep best, but we also want to diversify.
        # For simplicity, always update routes to current (like simulated annealing with T=0)
        routes = [list(r) for r in current_routes]

    return best_routes
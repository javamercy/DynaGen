import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def objective(routes):
        return max(route_distance(r) for r in routes)

    # --- Initial construction with minimax insertion (deterministic order) ---
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    for node in customers:
        best_max = float('inf')
        best_total = float('inf')
        best_route = None
        best_pos = None
        for r in range(truck_count):
            route = routes[r]
            for pos in range(1, len(route)):
                new_route = route[:pos] + [node] + route[pos:]
                new_dist = route_distance(new_route)
                current_max = 0.0
                for rr in range(truck_count):
                    if rr == r:
                        d = new_dist
                    else:
                        d = route_distance(routes[rr])
                    if d > current_max:
                        current_max = d
                if current_max < best_max or (current_max == best_max and new_dist < best_total):
                    best_max = current_max
                    best_total = new_dist
                    best_route = r
                    best_pos = pos
        routes[best_route].insert(best_pos, node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)

    # --- Ruin-and-recreate loop ---
    max_iter = min(20, n * 2)
    for _ in range(max_iter):
        # Compute edge contribution for each customer
        edges = []
        for r, route in enumerate(routes):
            if len(route) <= 2:
                continue
            for idx in range(1, len(route)-1):
                node = route[idx]
                prev = route[idx-1]
                nxt = route[idx+1]
                ec = dist[prev][node] + dist[node][nxt]
                edges.append((ec, node, r, idx))
        edges.sort(reverse=True, key=lambda x: x[0])
        remove_count = max(1, int(0.3 * (n-1)))
        to_remove = []
        selected = set()
        for ec, node, r, idx in edges:
            if node in selected:
                continue
            selected.add(node)
            to_remove.append((node, r, idx))
            if len(selected) == remove_count:
                break

        # Remove in reverse order by route and index
        to_remove.sort(key=lambda x: (x[1], x[2]), reverse=True)
        removed = []
        for node, r, idx in to_remove:
            route = routes[r]
            if idx < 1 or idx >= len(route)-1:
                continue
            if route[idx] == node:
                routes[r] = route[:idx] + route[idx+1:]
                if len(routes[r]) < 2:
                    routes[r] = [0, 0]
                removed.append(node)
        random.shuffle(removed)

        # Reinsert removed customers
        for node in removed:
            best_max = float('inf')
            best_total = float('inf')
            best_route = None
            best_pos = None
            for r in range(truck_count):
                route = routes[r]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [node] + route[pos:]
                    new_dist = route_distance(new_route)
                    current_max = 0.0
                    for rr in range(truck_count):
                        if rr == r:
                            d = new_dist
                        else:
                            d = route_distance(routes[rr])
                        if d > current_max:
                            current_max = d
                    if current_max < best_max or (current_max == best_max and new_dist < best_total):
                        best_max = current_max
                        best_total = new_dist
                        best_route = r
                        best_pos = pos
            routes[best_route].insert(best_pos, node)

        new_obj = objective(routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    # --- Final 2-opt improvement on best solution ---
    for r in range(truck_count):
        route = best_routes[r]
        if len(route) <= 3:
            continue
        improved = True
        local_iter = 0
        while improved and local_iter < 10:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    old_dist = route_distance(route)
                    if new_dist < old_dist:
                        best_routes[r] = new_route
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
            local_iter += 1

    report_best_vrp(best_routes)
    return best_routes
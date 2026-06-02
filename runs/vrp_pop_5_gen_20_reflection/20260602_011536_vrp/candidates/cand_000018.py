import numpy as np
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

    # Initial solution using minimax construction
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
                    # new distance for route r after insertion
                    new_dist = 0
                    prev = route[0]
                    for k in range(1, len(route)):
                        if k == pos:
                            new_dist += dist[prev][node]
                            prev = node
                        new_dist += dist[prev][route[k]]
                        prev = route[k]
                    # compute current max across all routes with this insertion
                    current_max = 0
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
                        best_node = node
                        best_route = r
                        best_pos = pos
        routes[best_route].insert(best_pos, best_node)
        unassigned.remove(best_node)

    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)

    # Adaptive ruin and recreate
    max_iter = min(20, n * 2)
    all_customers = list(range(1, n))
    for _ in range(max_iter):
        current_routes = [list(r) for r in routes]
        current_obj = objective(current_routes)

        # Determine which route is the max (or tied)
        route_dists = [route_distance(r) for r in routes]
        max_dist = max(route_dists)
        # Identify customers in routes with distance close to max
        target_routes = [i for i, d in enumerate(route_dists) if d >= max_dist * 0.9]
        # Remove a fraction of customers, favoring those in target routes
        remove_candidates = []
        for r_idx, route in enumerate(routes):
            for node in route[1:-1]:
                if r_idx in target_routes:
                    remove_candidates.append((node, 1.0))  # higher weight
                else:
                    remove_candidates.append((node, 0.5))
        # Weighted random selection
        weights = [w for _, w in remove_candidates]
        nodes = [n for n, _ in remove_candidates]
        # Determine removal count (20-40%)
        remove_count = max(1, int(random.uniform(0.2, 0.4) * (n-1)))
        # Sample without replacement
        chosen = random.choices(nodes, weights=weights, k=remove_count) if len(nodes) >= remove_count else nodes[:]
        to_remove = set(chosen[:remove_count])
        # Remove customers
        removed_list = []
        for node in list(to_remove):
            for r_idx, route in enumerate(routes):
                if node in route:
                    pos = route.index(node)
                    routes[r_idx] = route[:pos] + route[pos+1:]
                    if len(routes[r_idx]) < 2:
                        routes[r_idx] = [0, 0]
                    removed_list.append(node)
                    break
        # Reconstruct with random order
        random.shuffle(removed_list)
        unassigned = removed_list
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
                        new_dist = 0
                        prev = route[0]
                        for k in range(1, len(route)):
                            if k == pos:
                                new_dist += dist[prev][node]
                                prev = node
                            new_dist += dist[prev][route[k]]
                            prev = route[k]
                        current_max = 0
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
                            best_node = node
                            best_route = r
                            best_pos = pos
            routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)

        # Apply 2-opt on each route (limited iterations)
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            improved = True
            local_iter = 0
            while improved and local_iter < 10:
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_distance(new_route) < route_distance(route):
                            routes[r_idx] = new_route
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                local_iter += 1

        new_obj = objective(routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        # Always accept new solution (exploration)
    return best_routes
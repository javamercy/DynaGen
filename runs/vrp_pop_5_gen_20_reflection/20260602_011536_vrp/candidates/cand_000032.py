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

    # Initial minimax construction
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
                    # compute new distance for route r after insertion
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

    # Ruin and recreate with simulated annealing
    max_iter = min(20, n * 2)
    temperature = 0.1 * best_obj
    cooling_rate = 0.99
    current_routes = [list(r) for r in routes]
    current_obj = best_obj
    for iteration in range(max_iter):
        # Ruin: remove 20-40% customers biased toward high-cost routes
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        target_routes = [i for i, d in enumerate(route_dists) if d >= max_dist * 0.9]
        remove_candidates = []
        for r_idx, route in enumerate(current_routes):
            for node in route[1:-1]:
                weight = 1.0 if r_idx in target_routes else 0.3
                remove_candidates.append((node, weight))
        if not remove_candidates:
            break
        remove_count = max(1, int(random.uniform(0.2, 0.4) * (n-1)))
        nodes = [n for n, _ in remove_candidates]
        weights = [w for _, w in remove_candidates]
        chosen = random.choices(nodes, weights=weights, k=min(remove_count, len(nodes)))
        to_remove = set(chosen)
        removed_list = []
        for node in to_remove:
            for r_idx, route in enumerate(current_routes):
                if node in route:
                    pos = route.index(node)
                    current_routes[r_idx] = route[:pos] + route[pos+1:]
                    if len(current_routes[r_idx]) < 2:
                        current_routes[r_idx] = [0, 0]
                    removed_list.append(node)
                    break
        random.shuffle(removed_list)
        # Reconstruct with balanced insertion
        unassigned = removed_list
        while unassigned:
            best_max = float('inf')
            best_route_idx = None
            best_pos = None
            best_node = None
            best_route_dist = float('inf')
            for node in unassigned:
                for r in range(truck_count):
                    route = current_routes[r]
                    current_route_dist = route_distance(route)
                    for pos in range(1, len(route)):
                        # compute new route distance if node inserted at pos
                        new_route_dist = 0
                        prev = route[0]
                        for k in range(1, len(route)):
                            if k == pos:
                                new_route_dist += dist[prev][node]
                                prev = node
                            new_route_dist += dist[prev][route[k]]
                            prev = route[k]
                        # compute new max distance across all routes
                        other_max = max(route_distance(current_routes[rr]) for rr in range(truck_count) if rr != r)
                        new_max = max(new_route_dist, other_max)
                        # Tie-breaking: prefer lower current route distance to balance
                        if new_max < best_max or (new_max == best_max and current_route_dist < best_route_dist):
                            best_max = new_max
                            best_route_dist = current_route_dist
                            best_route_idx = r
                            best_pos = pos
                            best_node = node
            current_routes[best_route_idx].insert(best_pos, best_node)
            unassigned.remove(best_node)
        # Apply 2-opt on each route
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
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
                            current_routes[r_idx] = new_route
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                local_iter += 1
        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
        else:
            # Simulated annealing acceptance
            delta = new_obj - current_obj
            if delta > 0:
                accept_prob = math.exp(-delta / temperature)
                if random.random() < accept_prob:
                    pass  # accept worse solution
                else:
                    # revert to previous current_routes
                    current_routes = [list(r) for r in best_routes]  # revert to best
                    current_obj = best_obj
            else:
                current_obj = new_obj
        temperature *= cooling_rate
        if temperature < 1e-8:
            temperature = 1e-8
    return best_routes
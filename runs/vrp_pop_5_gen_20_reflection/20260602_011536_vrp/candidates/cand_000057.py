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
                    if new_max < best_max or (new_max == best_max and new_route_dist < best_total):
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

    max_iter = 20
    # Tabu list for recently removed customers to preserve diversity
    tabu_removed = set()
    tabu_tenure = 3
    # Iterate
    for it in range(max_iter):
        current_routes = [list(r) for r in best_routes]
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        # Find longest route indices (within 1% tolerance)
        longest_indices = [i for i, d in enumerate(route_dists) if d >= max_dist * 0.99]
        removed_list = []
        tabu_this_iter = set()
        for r_idx in longest_indices:
            route = current_routes[r_idx]
            if len(route) <= 2:
                continue
            # Fraction of customers to remove
            frac = random.uniform(0.3, 0.5)
            remove_count = max(1, int(frac * (len(route)-2)))
            if remove_count >= len(route)-2:
                remove_count = len(route)-2
            customers = route[1:-1]
            if len(customers) == 0:
                continue
            # Exclude tabu customers, but if all are tabu, ignore tabu
            eligible = [c for c in customers if c not in tabu_removed]
            if len(eligible) < remove_count:
                eligible = customers
            selected = random.sample(eligible, min(remove_count, len(eligible)))
            removed_list.extend(selected)
            tabu_this_iter.update(selected)
            # Remove selected from route
            new_route = [route[0]]
            for node in route[1:-1]:
                if node not in selected:
                    new_route.append(node)
            new_route.append(0)
            current_routes[r_idx] = new_route
            if len(current_routes[r_idx]) < 2:
                current_routes[r_idx] = [0, 0]
        # Update tabu list
        for c in tabu_this_iter:
            tabu_removed.add(c)
        # Limit tabu list size
        if len(tabu_removed) > tabu_tenure * (truck_count * 5):  # heuristic limit
            # Remove oldest entries (approximate: clear all and add recent)
            tabu_removed = set(list(tabu_removed)[-tabu_tenure * truck_count:])
        random.shuffle(removed_list)

        # Reconstruct with minimax insertion (simplified: only max then total distance)
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
                        new_route_dist = route_distance(new_route)
                        new_max = max(route_distance(current_routes[rr]) if rr != r else new_route_dist for rr in range(truck_count))
                        if new_max < best_max or (new_max == best_max and new_route_dist < best_total):
                            best_max = new_max
                            best_total = new_route_dist
                            best_node = node
                            best_route = r
                            best_pos = pos
            current_routes[best_route].insert(best_pos, best_node)
            unassigned.remove(best_node)

        # Intensified local search: only 2-opt on longest route
        route_dists = [route_distance(r) for r in current_routes]
        max_dist = max(route_dists)
        longest_indices = [i for i, d in enumerate(route_dists) if d >= max_dist * 0.99]
        for r_idx in longest_indices:
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            improved = True
            for _ in range(5):  # limited passes
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_distance(new_route) < route_distance(route):
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            current_routes[r_idx] = route

        new_obj = objective(current_routes)
        if new_obj < best_obj:
            best_obj = new_obj
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
        else:
            # Optionally break if no improvement (optional)
            pass

    return best_routes
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

    # Initial solution via minimax construction
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

    max_iter = min(30, 2 * n)
    remove_frac = 0.3
    for iteration in range(max_iter):
        current_routes = [list(r) for r in routes]
        # Random removal
        remove_count = max(1, int(remove_frac * (n - 1)))
        all_customers = list(range(1, n))
        random.shuffle(all_customers)
        to_remove = set(all_customers[:remove_count])

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
        unassigned = removed_list
        while unassigned:
            best_candidates = []
            best_max = float('inf')
            best_total = float('inf')
            for node in unassigned:
                for r in range(truck_count):
                    route = current_routes[r]
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
                                d = route_distance(current_routes[rr])
                            if d > current_max:
                                current_max = d
                        if current_max < best_max:
                            best_max = current_max
                            best_total = new_dist
                            best_candidates = [(node, r, pos)]
                        elif current_max == best_max:
                            if new_dist < best_total:
                                best_total = new_dist
                                best_candidates = [(node, r, pos)]
                            elif new_dist == best_total:
                                best_candidates.append((node, r, pos))
            if not best_candidates:
                break
            chosen = random.choice(best_candidates)
            node, best_route, best_pos = chosen
            current_routes[best_route].insert(best_pos, node)
            unassigned.remove(node)

        # Intra-route 2-opt (limited iterations)
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            max_iter_2opt = min(5, len(route))
            for _ in range(max_iter_2opt):
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_d = route_distance(new_route)
                        old_d = route_distance(route)
                        if new_d < old_d:
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
            # Accept with probability 0 (no simulated annealing)
            continue
        routes = current_routes

    return best_routes
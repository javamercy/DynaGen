import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_insertion_cost(c, route, current_dist):
        # returns (new_route_dist, best_pos) for best insertion position in route
        best = float('inf')
        best_pos = -1
        for pos in range(1, len(route)):
            pred = route[pos-1]
            succ = route[pos]
            new_dist = current_dist - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
            if new_dist < best:
                best = new_dist
                best_pos = pos
        return best, best_pos

    def best_insertion_with_second(c, routes, route_dists):
        # returns (best_new_max, best_route_idx, best_pos, second_best_new_max)
        best_new_max = float('inf')
        best_route = -1
        best_pos = -1
        second_new_max = float('inf')
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            new_route_dist, pos = compute_insertion_cost(c, route, route_dists[r_idx])
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            new_max = max(other_max, new_route_dist)
            if new_max < best_new_max:
                second_new_max = best_new_max
                best_new_max = new_max
                best_route = r_idx
                best_pos = pos
            elif new_max < second_new_max:
                second_new_max = new_max
        return best_new_max, best_route, best_pos, second_new_max

    def construct_routes():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        while unassigned:
            candidates = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion_with_second(c, routes, route_dists)
                if best_route == -1:
                    continue
                # regret = difference between second best and best (larger is better)
                if second_new_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_new_max - best_new_max
                candidates.append((regret, best_new_max, c, best_route, best_pos))
            if not candidates:
                break
            # select customer with highest regret; tie-break by lower best_new_max
            candidates.sort(key=lambda x: (-x[0], x[1]))
            _, _, c, best_route, best_pos = candidates[0]
            routes[best_route].insert(best_pos, c)
            route_dists[best_route] = route_dist(routes[best_route])
            unassigned.remove(c)
        return routes, route_dists

    def intra_2opt(routes, route_dists, affected):
        for r_idx in affected:
            improved = True
            while improved:
                improved = False
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
                            route_dists[r_idx] = route_dist(route)
                            break
                    if improved:
                        break
        return routes, route_dists

    def inter_relocate(routes, route_dists):
        best_move = None
        best_new_max = max(route_dists)
        for from_idx in range(truck_count):
            from_route = routes[from_idx]
            for i in range(1, len(from_route)-1):
                c = from_route[i]
                pred = from_route[i-1]
                succ = from_route[i+1]
                new_from_dist = route_dists[from_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
                for to_idx in range(truck_count):
                    if to_idx == from_idx:
                        continue
                    to_route = routes[to_idx]
                    for pos in range(1, len(to_route)):
                        pred_o = to_route[pos-1]
                        succ_o = to_route[pos]
                        new_to_dist = route_dists[to_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                        other_max = 0.0
                        for j, d in enumerate(route_dists):
                            if j != from_idx and j != to_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_from_dist, new_to_dist)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_move = (from_idx, i, to_idx, pos, new_from_dist, new_to_dist)
        return best_move, best_new_max

    # Construction
    routes, route_dists = construct_routes()
    # Improvement loop
    max_iter = 50
    for _ in range(max_iter):
        # Intra 2-opt on all routes
        routes, route_dists = intra_2opt(routes, route_dists, list(range(truck_count)))
        # Inter-relocate
        best_move, best_new_max = inter_relocate(routes, route_dists)
        if best_move is None:
            break
        from_idx, i, to_idx, pos, new_from_dist, new_to_dist = best_move
        c = routes[from_idx].pop(i)
        routes[to_idx].insert(pos, c)
        route_dists[from_idx] = new_from_dist
        route_dists[to_idx] = new_to_dist
        routes, route_dists = intra_2opt(routes, route_dists, [from_idx, to_idx])
        report_best_vrp(routes)
    # Final 2-opt
    routes, route_dists = intra_2opt(routes, route_dists, list(range(truck_count)))
    return routes
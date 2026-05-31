import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    n_cust = n - 1
    if truck_count >= n_cust:
        routes = [[0,0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        report_best_vrp(routes)
        return routes

    # Initialize empty routes
    routes = [[0,0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)

    def best_insertion(cust, routes, route_dists):
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = max([d for j,d in enumerate(route_dists) if j != r_idx], default=0.0)
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, cust] + distance_matrix[cust, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    # Regret-2 construction
    while unassigned:
        candidates = []
        for cust in unassigned:
            best_new_max, best_r, best_p, second_new_max = best_insertion(cust, routes, route_dists)
            if best_r == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            candidates.append((-regret, cust, best_r, best_p, best_new_max))
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, cust, best_r, best_p, new_max = candidates[0]
        route = routes[best_r]
        route.insert(best_p, cust)
        route_dists[best_r] = route_distance(route)
        unassigned.remove(cust)
        report_best_vrp(routes)

    best_routes = [r[:] for r in routes]
    best_max = max_route_distance(best_routes)

    # Intra-route 2-opt improvement
    def two_opt(route):
        if len(route) <= 2:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old_edges = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new_edges = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new_edges < old_edges - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        break
                if improved:
                    break
        return route

    for r_idx in range(truck_count):
        routes[r_idx] = two_opt(routes[r_idx])
        route_dists[r_idx] = route_distance(routes[r_idx])
    current_max = max_route_distance(routes)
    if current_max < best_max - 1e-12:
        best_routes = [r[:] for r in routes]
        best_max = current_max
        report_best_vrp(best_routes)

    # Inter-route best-improvement local search
    max_iter = n * truck_count
    for _ in range(max_iter):
        max_dist_val = max(route_dists)
        max_idx = route_dists.index(max_dist_val)
        best_move = None
        best_new_max = max_dist_val
        # Consider relocate moves from max route
        for pos in range(1, len(routes[max_idx])-1):
            cust = routes[max_idx][pos]
            # Remove cust from max route
            new_route_max = routes[max_idx][:pos] + routes[max_idx][pos+1:]
            new_dist_max = route_distance(new_route_max)
            for to_idx in range(truck_count):
                if to_idx == max_idx:
                    continue
                other_route = routes[to_idx]
                for insert_pos in range(1, len(other_route)):
                    new_other = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                    new_dist_other = route_distance(new_other)
                    other_max_val = max([d for j,d in enumerate(route_dists) if j != max_idx and j != to_idx], default=0.0)
                    new_max_val = max(other_max_val, new_dist_max, new_dist_other)
                    if new_max_val < best_new_max - 1e-12 or (abs(new_max_val - best_new_max) < 1e-12 and (max_idx < best_move[0] or (max_idx == best_move[0] and pos < best_move[1]))):
                        best_new_max = new_max_val
                        best_move = ('relocate', max_idx, pos, to_idx, insert_pos, new_route_max, new_dist_max, new_other, new_dist_other)
        # Consider exchange moves between max route and another route
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            for p1 in range(1, len(routes[max_idx])-1):
                for p2 in range(1, len(routes[other_idx])-1):
                    cust1 = routes[max_idx][p1]
                    cust2 = routes[other_idx][p2]
                    # Swap
                    new_max_route = routes[max_idx][:]
                    new_other_route = routes[other_idx][:]
                    new_max_route[p1] = cust2
                    new_other_route[p2] = cust1
                    new_dist_max = route_distance(new_max_route)
                    new_dist_other = route_distance(new_other_route)
                    other_max_val = max([d for j,d in enumerate(route_dists) if j != max_idx and j != other_idx], default=0.0)
                    new_max_val = max(other_max_val, new_dist_max, new_dist_other)
                    if new_max_val < best_new_max - 1e-12 or (abs(new_max_val - best_new_max) < 1e-12 and (max_idx < best_move[0] or (max_idx == best_move[0] and p1 < best_move[1]))):
                        best_new_max = new_max_val
                        best_move = ('exchange', max_idx, p1, other_idx, p2, new_max_route, new_dist_max, new_other_route, new_dist_other)

        if best_move is None or best_new_max >= max_dist_val - 1e-12:
            break
        # Apply best move
        if best_move[0] == 'relocate':
            _, max_idx, pos, to_idx, insert_pos, new_max_route, new_dist_max, new_other_route, new_dist_other = best_move
            routes[max_idx] = new_max_route
            routes[to_idx] = new_other_route
            route_dists[max_idx] = new_dist_max
            route_dists[to_idx] = new_dist_other
        else:
            _, max_idx, p1, other_idx, p2, new_max_route, new_dist_max, new_other_route, new_dist_other = best_move
            routes[max_idx] = new_max_route
            routes[other_idx] = new_other_route
            route_dists[max_idx] = new_dist_max
            route_dists[other_idx] = new_dist_other
        # Apply 2-opt on affected routes
        routes[max_idx] = two_opt(routes[max_idx])
        route_dists[max_idx] = route_distance(routes[max_idx])
        routes[other_idx if best_move[0]=='relocate' else other_idx] = two_opt(routes[other_idx])
        route_dists[other_idx if best_move[0]=='relocate' else other_idx] = route_distance(routes[other_idx])
        current_max = max_route_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    return best_routes
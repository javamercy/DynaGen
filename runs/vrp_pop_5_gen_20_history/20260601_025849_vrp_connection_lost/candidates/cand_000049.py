import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))

    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    # Regret-2 construction
    while unassigned:
        best_info = {}
        for c in unassigned:
            best = float('inf')
            second = float('inf')
            best_r = -1
            best_p = -1
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                    if cost < best:
                        second = best
                        best = cost
                        best_r = r_idx
                        best_p = i + 1
                    elif cost < second:
                        second = cost
            best_info[c] = (best, second, best_r, best_p)

        candidates = []
        for c, (best, second, r_idx, pos) in best_info.items():
            regret = second - best if second != float('inf') else float('inf')
            new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
            new_route_dist = route_dist(new_route)
            other_max = 0.0
            if truck_count > 1:
                other_max = max(route_dist(r) for i, r in enumerate(routes) if i != r_idx)
            new_max = max(new_route_dist, other_max)
            candidates.append((-regret, new_max, c, r_idx, pos))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_c, chosen_r, chosen_p = candidates[0]
        routes[chosen_r].insert(chosen_p, chosen_c)
        unassigned.remove(chosen_c)

    report_best_vrp(routes)
    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)

    # Pre-improvement: 2-opt on each route individually
    for r_idx, route in enumerate(routes):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_dist(new_route) < route_dist(route):
                        routes[r_idx] = new_route
                        route = new_route
                        improved = True
                        current_max = max_dist(routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break

    # Improvement loop with adaptive operator schedule
    max_iter = n * n
    operators = ['relocate', 'swap', '2opt']
    op_idx = 0
    for _ in range(max_iter):
        improved = False
        current_max = max_dist(routes)
        # Find longest routes, pick one with most customers among ties
        max_dist_val = current_max
        candidates_longest = [(r_idx, len(routes[r_idx]), route_dist(routes[r_idx])) for r_idx in range(truck_count) if route_dist(routes[r_idx]) == max_dist_val]
        if not candidates_longest:
            break
        candidates_longest.sort(key=lambda x: (-x[1], x[2]))
        r_idx = candidates_longest[0][0]
        route = routes[r_idx]

        if operators[op_idx] == 'relocate':
            # Relocate from longest route
            for pos in range(1, len(route)-1):
                cust = route[pos]
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        continue
                    for other_pos in range(1, len(other_route)):
                        new_self = route[:pos] + route[pos+1:]
                        new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                        new_routes = [list(r) for r in routes]
                        new_routes[r_idx] = new_self
                        new_routes[other_idx] = new_other
                        new_max = max_dist(new_routes)
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in new_routes]
                            routes = new_routes
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
        elif operators[op_idx] == 'swap':
            # Inter-route swap
            for pos1 in range(1, len(route)-1):
                cust1 = route[pos1]
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        continue
                    if len(other_route) <= 2:
                        continue
                    for pos2 in range(1, len(other_route)-1):
                        cust2 = other_route[pos2]
                        new_route1 = route[:pos1] + [cust2] + route[pos1+1:]
                        new_route2 = other_route[:pos2] + [cust1] + other_route[pos2+1:]
                        new_routes = [list(r) for r in routes]
                        new_routes[r_idx] = new_route1
                        new_routes[other_idx] = new_route2
                        new_max = max_dist(new_routes)
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in new_routes]
                            routes = new_routes
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
        else:  # 2-opt
            # 2-opt on all routes in order of decreasing distance
            routes_sorted = sorted(enumerate(routes), key=lambda x: -route_dist(x[1]))
            for idx, r in routes_sorted:
                if len(r) <= 3:
                    continue
                for i in range(1, len(r)-2):
                    for j in range(i+1, len(r)-1):
                        new_route = r[:i] + r[i:j+1][::-1] + r[j+1:]
                        if route_dist(new_route) < route_dist(r):
                            routes[idx] = new_route
                            improved = True
                            current_max = max_dist(routes)
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break

        if improved:
            # Continue with same operator
            op_idx = op_idx  # unchanged
        else:
            op_idx = (op_idx + 1) % 3
            # If all three operators failed consecutively, break
            if op_idx == 0:
                break

    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    return final_routes
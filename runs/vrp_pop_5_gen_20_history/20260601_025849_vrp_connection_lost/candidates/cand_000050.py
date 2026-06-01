import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix

    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    # Construction: regret-2 with simple tie-breaking (first max regret)
    def construct():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            best_info = {}  # c -> (best_cost, second_cost, best_route_idx, best_pos)
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

            # Choose customer with max regret, tie-break by largest best cost (worsens overall? but simple)
            chosen_c = None
            best_regret = -1.0
            best_best_cost = float('inf')
            for c, (best, second, r_idx, pos) in best_info.items():
                regret = second - best if second != float('inf') else float('inf')
                if regret > best_regret or (regret == best_regret and best > best_best_cost):
                    best_regret = regret
                    best_best_cost = best
                    chosen_c = c
                    chosen_r = r_idx
                    chosen_p = pos
            routes[chosen_r].insert(chosen_p, chosen_c)
            unassigned.remove(chosen_c)
        return routes

    # Initial construction
    routes = construct()
    report_best_vrp(routes)
    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)

    # Improvement parameters
    max_iter = n * n
    stagnation_limit = max(10, (n-1) // 10)
    consecutive_no_improvement = 0

    for _ in range(max_iter):
        improved = False
        current_max = max_dist(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_dist(r) == current_max]
        if not longest_indices:
            break
        r_idx = longest_indices[0]
        route = routes[r_idx]

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
        if improved:
            consecutive_no_improvement = 0
            continue

        # 2-opt on each route
        for r_idx2, route2 in enumerate(routes):
            if len(route2) <= 3:
                continue
            improved_inner = False
            for i in range(1, len(route2)-2):
                for j in range(i+1, len(route2)-1):
                    new_route = route2[:i] + route2[i:j+1][::-1] + route2[j+1:]
                    if route_dist(new_route) < route_dist(route2):
                        routes[r_idx2] = new_route
                        improved_inner = True
                        current_max = max_dist(routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        break
                if improved_inner:
                    break
            if improved_inner:
                improved = True
                consecutive_no_improvement = 0
                break

        if not improved:
            consecutive_no_improvement += 1
            if consecutive_no_improvement >= stagnation_limit:
                break

    # Format output
    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    return final_routes
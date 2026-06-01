import numpy as np
import math
import heapq
import itertools
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix

    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    total_customers = n - 1

    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    # Adaptive regret construction
    while unassigned:
        remaining = len(unassigned)
        # Use regret-2 if more than 20% remain, else regret-3
        regret_k = 2 if remaining > 0.2 * total_customers else 3

        best_info = {}
        for c in unassigned:
            costs = []
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                    costs.append((cost, r_idx, i + 1))
            costs.sort(key=lambda x: x[0])
            best_cost = costs[0][0]
            second_cost = costs[1][0] if len(costs) > 1 else float('inf')
            third_cost = costs[2][0] if len(costs) > 2 else float('inf')
            if regret_k == 2:
                regret = second_cost - best_cost if second_cost != float('inf') else float('inf')
            else:
                regret = third_cost - best_cost if third_cost != float('inf') else float('inf')
            best_r, best_p = costs[0][1], costs[0][2]
            # compute new max if inserted into best route
            new_route = routes[best_r][:best_p] + [c] + routes[best_r][best_p:]
            new_route_dist = route_dist(new_route)
            other_max = max(route_dist(r) for i, r in enumerate(routes) if i != best_r) if truck_count > 1 else 0.0
            new_max = max(new_route_dist, other_max)
            candidates.append((-regret if regret != float('inf') else float('-inf'), new_max, c, best_r, best_p))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_c, chosen_r, chosen_p = candidates[0]
        routes[chosen_r].insert(chosen_p, chosen_c)
        unassigned.remove(chosen_c)

    report_best_vrp(routes)
    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)

    # Improvement: bounded iteration
    max_iter = n * truck_count * 2
    for _ in range(max_iter):
        improved = False
        # 2-opt for each route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_dist(new_route) < route_dist(route):
                        routes[r_idx] = new_route
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
            continue
        # Relocate from longest route with limit on attempts
        max_val = max_dist(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_dist(r) == max_val]
        for r_idx in longest_indices:
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            # Limit number of relocate attempts to 5 * (len(route)-2) * (truck_count-1)
            max_attempts = 5 * (len(route)-2) * (truck_count-1)
            attempts = 0
            for pos in range(1, len(route)-1):
                cust = route[pos]
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        continue
                    for ins_pos in range(1, len(other_route)):
                        if attempts >= max_attempts:
                            break
                        attempts += 1
                        new_self = route[:pos] + route[pos+1:]
                        new_other = other_route[:ins_pos] + [cust] + other_route[ins_pos:]
                        new_routes = [list(r) for i, r in enumerate(routes)]
                        new_routes[r_idx] = new_self
                        new_routes[other_idx] = new_other
                        new_max = max_dist(new_routes)
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in new_routes]
                            routes = new_routes
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved or attempts >= max_attempts:
                        break
                if improved or attempts >= max_attempts:
                    break
            if improved:
                break
        if not improved:
            break

    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    return final_routes
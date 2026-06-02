import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    best_routes = [[0, 0] for _ in range(truck_count)]
    best_max = float('inf')
    
    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        dists = [route_distance(r) for r in routes]
        cur_max = max(dists)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [list(r) for r in routes]
    
    # Regret-k insertion with k=2
    def regret_insertion(unassigned, routes, dists):
        while unassigned:
            options_per_customer = {}
            for c in unassigned:
                options = []
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for i in range(1, len(route)):
                        new_dist = dists[r_idx] - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                        other_dists = [dists[t] for t in range(truck_count) if t != r_idx]
                        new_max = new_dist if not other_dists else max(new_dist, max(other_dists))
                        options.append((new_max, r_idx, i, new_dist))
                options.sort(key=lambda x: x[0])
                options_per_customer[c] = options
            # Compute regret
            regret_list = []
            for c in unassigned:
                opts = options_per_customer[c]
                if len(opts) >= 2:
                    regret = opts[1][0] - opts[0][0]
                else:
                    regret = 0.0
                # Tie-breaker: farthest from depot first
                tie = distance_matrix[0, c]
                regret_list.append((-regret, -tie, c, opts[0]))
            regret_list.sort(key=lambda x: (x[0], x[1]))
            _, _, c, best_opt = regret_list[0]
            r_idx, i, new_d = best_opt[1], best_opt[2], best_opt[3]
            routes[r_idx].insert(i, c)
            dists[r_idx] = route_distance(routes[r_idx])
            unassigned.remove(c)
    
    # Construction
    routes = [[0, 0] for _ in range(truck_count)]
    dists = [0.0 for _ in range(truck_count)]
    unassigned = set(range(1, n))
    regret_insertion(unassigned, routes, dists)
    report_best_vrp(routes)
    
    # Intra-route 2-opt improvement
    def improve(routes, dists):
        improved = True
        max_iter = n * n
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for r_idx in range(truck_count):
                route = routes[r_idx]
                best_route = route[:]
                best_dist = dists[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
                if improved:
                    routes[r_idx] = best_route
                    dists[r_idx] = best_dist
                    new_max = max(dists)
                    if new_max < best_max:
                        report_best_vrp(routes)
                    break
        return routes, dists
    
    routes, dists = improve(routes, dists)
    
    # Restart: remove up to 10% from longest route and reinsert
    if n > 1:
        max_dist = max(dists)
        longest_idx = int(np.argmax(dists))
        longest_route = routes[longest_idx]
        if len(longest_route) > 3:
            remove_count = max(1, int(0.1 * (len(longest_route)-2)))
            removals = longest_route[1:-1][:remove_count]
            new_route = [0] + longest_route[1+remove_count:-1] + [0]
            routes[longest_idx] = new_route
            dists[longest_idx] = route_distance(new_route)
            unassigned = set(removals)
            regret_insertion(unassigned, routes, dists)
            routes, dists = improve(routes, dists)
            report_best_vrp(routes)
    
    return best_routes
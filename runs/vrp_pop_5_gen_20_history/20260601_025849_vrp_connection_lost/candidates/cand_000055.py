import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix

    def route_dist(route):
        if len(route) <= 1:
            return 0.0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def construct(randomized=False):
        routes = [[0, 0] for _ in range(truck_count)]
        customers = list(range(1, n))
        if randomized:
            random.shuffle(customers)
        unassigned = set(customers)
        while unassigned:
            best_regret = -float('inf')
            best_cust = None
            best_insert = None
            best_new_max = float('inf')
            for c in unassigned:
                best_cost = float('inf')
                second_cost = float('inf')
                best_route = -1
                best_pos = -1
                for r_idx, route in enumerate(routes):
                    for i in range(len(route) - 1):
                        delta = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                        if delta < best_cost:
                            second_cost = best_cost
                            best_cost = delta
                            best_route = r_idx
                            best_pos = i + 1
                        elif delta < second_cost:
                            second_cost = delta
                regret = second_cost - best_cost if second_cost != float('inf') else float('inf')
                new_route = routes[best_route][:best_pos] + [c] + routes[best_route][best_pos:]
                new_route_dist = route_dist(new_route)
                other_max = max(route_dist(r) for i, r in enumerate(routes) if i != best_route) if truck_count > 1 else 0
                new_max = max(new_route_dist, other_max)
                if (regret > best_regret or
                    (regret == best_regret and best_cost < best_insert[0]) or
                    (regret == best_regret and best_cost == best_insert[0] and new_max < best_new_max)):
                    best_regret = regret
                    best_cust = c
                    best_insert = (best_cost, best_route, best_pos)
                    best_new_max = new_max
            _, r_idx, pos = best_insert
            routes[r_idx].insert(pos, best_cust)
            unassigned.remove(best_cust)
        return routes

    def improve(routes):
        improved = True
        max_iter = n * 3
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # relocate
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route) - 1):
                    cust = route[pos]
                    new_route = route[:pos] + route[pos+1:]
                    for other_r_idx in range(truck_count):
                        other_route = routes[other_r_idx]
                        for other_pos in range(1, len(other_route) + 1):
                            new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                            new_routes = [list(r) for r in routes]
                            new_routes[r_idx] = new_route
                            new_routes[other_r_idx] = new_other
                            new_max = max_dist(new_routes)
                            if new_max < max_dist(routes):
                                routes = new_routes
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # inter-route swap
            for r1_idx in range(truck_count):
                route1 = routes[r1_idx]
                for pos1 in range(1, len(route1) - 1):
                    cust1 = route1[pos1]
                    for r2_idx in range(r1_idx + 1, truck_count):
                        route2 = routes[r2_idx]
                        for pos2 in range(1, len(route2) - 1):
                            cust2 = route2[pos2]
                            new_route1 = route1[:pos1] + [cust2] + route1[pos1+1:]
                            new_route2 = route2[:pos2] + [cust1] + route2[pos2+1:]
                            new_routes = [list(r) for r in routes]
                            new_routes[r1_idx] = new_route1
                            new_routes[r2_idx] = new_route2
                            new_max = max_dist(new_routes)
                            if new_max < max_dist(routes):
                                routes = new_routes
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    best_routes = None
    best_max = float('inf')
    for restart in range(10):
        routes = construct(randomized=(restart > 0))
        routes = improve(routes)
        current_max = max_dist(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    final = []
    for r in best_routes:
        if len(r) == 2:
            final.append([0, 0])
        else:
            final.append([0] + r[1:-1] + [0])
    return final
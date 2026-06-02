import numpy as np
from typing import List

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    n = distance_matrix.shape[0]
    if truck_count >= n:
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])
        for _ in range(truck_count - (n - 1)):
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) == 2:
            return distance_matrix[0, 0]
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i], route[i + 1]]
        return total

    def insertion_cost(route, customer, pos):
        prev = route[pos - 1]
        nxt = route[pos]
        return distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        return route

    # Construction: regret-2 insertion
    routes = [[0, 0] for _ in range(truck_count)]
    remaining = set(range(1, n))
    while remaining:
        best_customer = None
        best_regret = -1.0
        best_route_idx = -1
        best_pos = -1
        for c in sorted(remaining):
            best_cost = float('inf')
            second_best_cost = float('inf')
            best_route = -1
            best_pos_local = -1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    cost = insertion_cost(route, c, pos)
                    if cost < best_cost:
                        second_best_cost = best_cost
                        best_cost = cost
                        best_route = r_idx
                        best_pos_local = pos
                    elif cost < second_best_cost:
                        second_best_cost = cost
            if second_best_cost == float('inf'):
                regret = float('inf')
            else:
                regret = second_best_cost - best_cost
            if regret > best_regret:
                best_regret = regret
                best_customer = c
                best_route_idx = best_route
                best_pos = best_pos_local
        route = routes[best_route_idx]
        route.insert(best_pos, best_customer)
        remaining.remove(best_customer)

    # Initial 2-opt pass on all routes
    for i in range(truck_count):
        routes[i] = two_opt(routes[i])

    best_routes = [r[:] for r in routes]
    best_max = max_distance(routes)
    report_best_vrp(best_routes)

    # Improvement: move customers from longest to shortest
    max_iters = n * truck_count
    for _ in range(max_iters):
        lengths = [(route_distance(r), idx) for idx, r in enumerate(routes)]
        lengths.sort(reverse=True, key=lambda x: x[0])
        longest_idx = lengths[0][1]
        shortest_idx = lengths[-1][1]
        if lengths[0][0] == lengths[-1][0]:
            break
        longest_route = routes[longest_idx]
        shortest_route = routes[shortest_idx]
        # Iterate customers in order (excluding depots)
        custs_long = [c for c in longest_route[1:-1]]
        moved = False
        for cust in custs_long:
            # remove from longest
            new_long = [0] + [c for c in longest_route[1:-1] if c != cust] + [0]
            # find best insertion position in shortest
            best_p = -1
            best_inc = float('inf')
            for pos in range(1, len(shortest_route)):
                inc = insertion_cost(shortest_route, cust, pos)
                if inc < best_inc:
                    best_inc = inc
                    best_p = pos
            new_short = shortest_route[:best_p] + [cust] + shortest_route[best_p:]
            # Only apply 2-opt after move if it improves max
            # Compute max without 2-opt first
            new_routes = routes[:]
            new_routes[longest_idx] = new_long
            new_routes[shortest_idx] = new_short
            new_max = max_distance(new_routes)
            if new_max < best_max:
                # Apply 2-opt to both routes
                new_long = two_opt(new_long)
                new_short = two_opt(new_short)
                new_routes[longest_idx] = new_long
                new_routes[shortest_idx] = new_short
                new_max = max_distance(new_routes)
                if new_max < best_max:
                    routes = new_routes
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                    moved = True
                    break
        if not moved:
            break

    return best_routes
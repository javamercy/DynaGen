import numpy as np
from typing import List

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)

    def insert_customer(route, pos, cust):
        return route[:pos] + [cust] + route[pos:]

    # Greedy insertion (same as parent)
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = insert_customer(route, pos, cust)
                new_route_dist = route_distance(new_route)
                other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx) if truck_count > 1 else 0.0
                new_max = max(new_route_dist, other_max)
                if new_max < best_max or (new_max == best_max and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        route = routes[best_route_idx]
        routes[best_route_idx] = insert_customer(route, best_pos, cust)

    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)
    report_best_vrp(best_routes)  # report initial solution

    # Refinement: best-improvement local search (replacing parent's first-improvement)
    max_iter = n * n
    for _ in range(max_iter):
        improved = False
        current_max = max_route_distance(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_distance(r) == current_max]
        if not longest_indices:
            break
        r_idx = longest_indices[0]  # first longest route (deterministic)
        route = routes[r_idx]
        best_move = None
        best_new_max = current_max  # only consider strict improvement

        # Evaluate all relocations from longest route to other routes
        for pos in range(1, len(route)-1):
            cust = route[pos]
            for other_idx, other_route in enumerate(routes):
                if other_idx == r_idx:
                    continue
                for other_pos in range(1, len(other_route)):
                    new_other = insert_customer(other_route, other_pos, cust)
                    new_self = route[:pos] + route[pos+1:]
                    new_routes = [list(r) for r in routes]
                    new_routes[r_idx] = new_self
                    new_routes[other_idx] = new_other
                    new_max = max_route_distance(new_routes)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = ('reloc', r_idx, pos, cust, other_idx, other_pos)

        # Evaluate all swaps between longest route and other routes
        for pos in range(1, len(route)-1):
            cust1 = route[pos]
            for other_idx, other_route in enumerate(routes):
                if other_idx == r_idx:
                    continue
                for other_pos in range(1, len(other_route)-1):
                    cust2 = other_route[other_pos]
                    new_route1 = route[:pos] + [cust2] + route[pos+1:]
                    new_route2 = other_route[:other_pos] + [cust1] + other_route[other_pos+1:]
                    new_routes = [list(r) for r in routes]
                    new_routes[r_idx] = new_route1
                    new_routes[other_idx] = new_route2
                    new_max = max_route_distance(new_routes)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = ('swap', r_idx, pos, cust1, other_idx, other_pos, cust2)

        if best_move is None:
            break
        # Apply best move
        if best_move[0] == 'reloc':
            _, r_idx, pos, cust, other_idx, other_pos = best_move
            route = routes[r_idx]
            other_route = routes[other_idx]
            new_self = route[:pos] + route[pos+1:]
            new_other = insert_customer(other_route, other_pos, cust)
            routes[r_idx] = new_self
            routes[other_idx] = new_other
        else:  # swap
            _, r_idx, pos, cust1, other_idx, other_pos, cust2 = best_move
            route = routes[r_idx]
            other_route = routes[other_idx]
            new_route1 = route[:pos] + [cust2] + route[pos+1:]
            new_route2 = other_route[:other_pos] + [cust1] + other_route[other_pos+1:]
            routes[r_idx] = new_route1
            routes[other_idx] = new_route2

        new_max = max_route_distance(routes)
        if new_max < best_max:
            best_max = new_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        improved = True
        # Continue loop if improvement found; otherwise break (but we already break if no best_move)
    return best_routes
import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)

    # Greedy insertion minimizing max route distance
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route)
                # Compute max distance among all routes with this insertion
                other_max = 0.0
                for i, r in enumerate(routes):
                    if i != r_idx:
                        d = route_distance(r)
                        if d > other_max:
                            other_max = d
                new_max = max(new_dist, other_max)
                if new_max < best_max or (new_max == best_max and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        route = routes[best_route_idx]
        routes[best_route_idx] = route[:best_pos] + [cust] + route[best_pos:]

    # Report initial solution
    report_best_vrp([list(r) for r in routes])

    current_max = max_route_distance(routes)
    improved = True
    max_iter = n * truck_count  # finite bound
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        # Find longest route
        longest_idx = -1
        longest_dist = -1.0
        for i, r in enumerate(routes):
            d = route_distance(r)
            if d > longest_dist:
                longest_dist = d
                longest_idx = i
        longest_route = routes[longest_idx]
        if len(longest_route) <= 3:
            break
        # Try moving each customer from longest route (excluding depots)
        for pos in range(1, len(longest_route)-1):
            cust = longest_route[pos]
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx:
                    continue
                for other_pos in range(1, len(other_route)+1):  # allow insertion before depot
                    new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                    new_self = longest_route[:pos] + longest_route[pos+1:]
                    new_routes = list(routes)
                    new_routes[longest_idx] = new_self
                    new_routes[other_idx] = new_other
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < current_max:
                        # First improvement found
                        routes = new_routes
                        current_max = new_max
                        improved = True
                        report_best_vrp([list(r) for r in routes])
                        break
                if improved:
                    break
            if improved:
                break
        # If no improvement, loop ends
    # Return best routes (routes are already the best found)
    return routes
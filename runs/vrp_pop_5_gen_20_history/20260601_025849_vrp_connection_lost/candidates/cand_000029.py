import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
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

    # Greedy insertion (original deterministic tie-breaking)
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route)
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

    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)

    # Relocate improvement with adaptive early termination
    max_iter = n * n
    consecutive_no_improve = 0
    no_improve_limit = n  # adaptive threshold based on instance size
    for _ in range(max_iter):
        if consecutive_no_improve >= no_improve_limit:
            break
        current_max = max_route_distance(routes)
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
        improved = False
        # Try moving each customer from longest route to another route
        for pos in range(1, len(longest_route)-1):
            cust = longest_route[pos]
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx:
                    continue
                for other_pos in range(1, len(other_route)):
                    new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                    new_self = longest_route[:pos] + longest_route[pos+1:]
                    new_routes = list(routes)
                    new_routes[longest_idx] = new_self
                    new_routes[other_idx] = new_other
                    new_max = max_route_distance(new_routes)
                    if new_max < best_max:
                        best_max = new_max
                        best_routes = [list(r) for r in new_routes]
                        routes = new_routes
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    # Report best found solution
    # report_best_vrp(best_routes)  # Uncomment if harness expects this call
    return best_routes
import numpy as np

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Initialize truck_count empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    # Assign customers round-robin
    for idx, cust in enumerate(customers):
        route_idx = idx % truck_count
        routes[route_idx].insert(-1, cust)
    # Report initial solution
    report_best_vrp(routes)
    # Rebalancing improvement to minimize max route distance
    max_iter = len(customers) * truck_count
    for _ in range(max_iter):
        dists = [route_distance(r, distance_matrix) for r in routes]
        max_dist = max(dists)
        min_dist = min(dists)
        if max_dist == min_dist:
            break
        max_idx = dists.index(max_dist)
        best_improvement = 0
        best_move = None
        route_max = routes[max_idx]
        if len(route_max) <= 2:
            break
        for pos_cust in range(1, len(route_max)-1):
            customer = route_max[pos_cust]
            for other_idx, other_route in enumerate(routes):
                if other_idx == max_idx:
                    continue
                for insert_pos in range(1, len(other_route)):
                    new_route_max = route_max[:pos_cust] + route_max[pos_cust+1:]
                    new_other = other_route[:insert_pos] + [customer] + other_route[insert_pos:]
                    new_dists = dists.copy()
                    new_dists[max_idx] = route_distance(new_route_max, distance_matrix)
                    new_dists[other_idx] = route_distance(new_other, distance_matrix)
                    new_max = max(new_dists)
                    if new_max < max_dist - 1e-9:
                        improvement = max_dist - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (max_idx, pos_cust, other_idx, insert_pos)
        if best_move is not None:
            max_idx, pos_cust, other_idx, insert_pos = best_move
            customer = routes[max_idx][pos_cust]
            routes[max_idx] = routes[max_idx][:pos_cust] + routes[max_idx][pos_cust+1:]
            routes[other_idx] = routes[other_idx][:insert_pos] + [customer] + routes[other_idx][insert_pos:]
            report_best_vrp(routes)
        else:
            break
    report_best_vrp(routes)
    return routes